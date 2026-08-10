#ifndef READ_MTX_H
#define READ_MTX_H

#include <limits.h>

int read_mtx(char  * filename, int* m_add, int *n_add, int *nnzA_add,
             int **csrRowPtrA_add, int **csrColIdxA_add, VALUE_TYPE **csrValA_add)
{
    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;
    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    
    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;
    
    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
    {
        printf("Failed to open %s.\n", filename);
        return -1;
    }
    
    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        fclose(f);
        return -2;
    }
    
    if ( mm_is_complex( matcode ) )
    {
        printf("Sorry, data type 'COMPLEX' is not supported.\n");
        fclose(f);
        return -3;
    }
    
    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    
    if (ret_code != 0)
    {
        fprintf(stderr, "ERROR: could not parse matrix size in %s\n", filename);
        fclose(f);
        return -4;
    }

    // the row loops below count with int up to m inclusive, so m + 1 has to
    // stay representable
    if (m <= 0 || m == INT_MAX || n <= 0 || nnzA_mtx_report < 0)
    {
        fprintf(stderr, "ERROR: invalid matrix dimensions in %s\n", filename);
        fclose(f);
        return -7;
    }

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric = 1;
        //printf("input matrix is symmetric = true\n");
    }

    // a symmetric entry is mirrored into the row of its column index, so a
    // column index must address a valid row as well
    if (isSymmetric && m != n)
    {
        fprintf(stderr, "ERROR: symmetric matrix is not square in %s\n", filename);
        fclose(f);
        return -8;
    }

    size_t csrRowPtrA_size = ((size_t)m + 1) * sizeof(int);
    
    int *csrRowPtrA_counter = (int *)malloc(csrRowPtrA_size);
    
    int *csrRowIdxA_tmp = (int *)malloc((size_t)nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *)malloc((size_t)nnzA_mtx_report * sizeof(int));
    VALUE_TYPE *csrValA_tmp    = (VALUE_TYPE *)malloc((size_t)nnzA_mtx_report * sizeof(VALUE_TYPE));
    
    if(csrRowPtrA_counter == NULL ||
       (nnzA_mtx_report > 0 &&
        (csrRowIdxA_tmp == NULL || csrColIdxA_tmp == NULL || csrValA_tmp == NULL)))
    {
        fprintf(stderr, "ERROR: failed to allocate memory for %s\n", filename);
        fclose(f);
        free(csrRowPtrA_counter);
        free(csrRowIdxA_tmp);
        free(csrColIdxA_tmp);
        free(csrValA_tmp);
        return -2;
    }

    memset(csrRowPtrA_counter, 0, csrRowPtrA_size);
    
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
    
    //printf("222222\n");
    int i;
    for (i = 0; i < nnzA_mtx_report; i++)
    {
        int idxi = 0, idxj = 0;
        int ival = 0;
        double fval = 0.0;
        int returnvalue = 0;
        int expected_values = 0;
        
        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
            expected_values = 3;
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
            expected_values = 3;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
            expected_values = 2;
        }

        if (returnvalue != expected_values)
        {
            fprintf(stderr, "ERROR: malformed entry line in %s\n", filename);
            fclose(f);
            free(csrRowPtrA_counter);
            free(csrRowIdxA_tmp);
            free(csrColIdxA_tmp);
            free(csrValA_tmp);
            return -5;
        }

        if (idxi < 1 || idxi > m || idxj < 1 || idxj > n)
        {
            fprintf(stderr, "ERROR: entry index out of range in %s\n", filename);
            fclose(f);
            free(csrRowPtrA_counter);
            free(csrRowIdxA_tmp);
            free(csrColIdxA_tmp);
            free(csrValA_tmp);
            return -6;
        }
        
        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }
    
    fclose(f);
    
    if (isSymmetric)
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }
    
    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;
    
    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (i = 1; i <= m; i++)
    {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i-1];
        old_val = new_val;
    }
    
    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *)malloc(csrRowPtrA_size);
    csrColIdxA = (int *)malloc((size_t)nnzA * sizeof(int));
    csrValA    = (VALUE_TYPE *)malloc((size_t)nnzA * sizeof(VALUE_TYPE));

    if (csrRowPtrA == NULL || (nnzA > 0 && (csrColIdxA == NULL || csrValA == NULL)))
    {
        fprintf(stderr, "ERROR: failed to allocate memory for %s\n", filename);
        free(csrRowPtrA);
        free(csrColIdxA);
        free(csrValA);
        free(csrRowPtrA_counter);
        free(csrRowIdxA_tmp);
        free(csrColIdxA_tmp);
        free(csrValA_tmp);
        return -2;
    }
    
    memcpy(csrRowPtrA, csrRowPtrA_counter, csrRowPtrA_size);
    memset(csrRowPtrA_counter, 0, csrRowPtrA_size);
    
    if (isSymmetric)
    {
        for ( i = 0; i < nnzA_mtx_report; i++)
        {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
                
                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    }
    else
    {
        for (i = 0; i < nnzA_mtx_report; i++)
        {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }
    
    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);
    
    //printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
    *m_add=m;
    *n_add=n;
    *nnzA_add=nnzA;
    *csrColIdxA_add=csrColIdxA;
    *csrValA_add=csrValA;
    *csrRowPtrA_add=csrRowPtrA;
    
    
    return 0;
}


int change2tran(int m, int nnzA,int *csrRowPtrA, int *csrColIdxA,
                 VALUE_TYPE *csrValA, int *nnzL_add, int **csrRowPtrL_tmp_add,
                 int **csrColIdxL_tmp_add, VALUE_TYPE **csrValL_tmp_add)
{
    int nnzL = 0;
    // besides the strictly lower entries of A, every row contributes a unit
    // diagonal that A itself does not necessarily store
    size_t max_nnzL = (size_t)nnzA + (size_t)m;

    int *csrRowPtrL_tmp = (int *)malloc(((size_t)m + 1) * sizeof(int));
    int *csrColIdxL_tmp = (int *)malloc(max_nnzL * sizeof(int));
    VALUE_TYPE *csrValL_tmp    = (VALUE_TYPE *)malloc(max_nnzL * sizeof(VALUE_TYPE));

    if (csrRowPtrL_tmp==NULL || csrColIdxL_tmp==NULL || csrValL_tmp==NULL)
    {
        fprintf(stderr, "ERROR: failed to allocate memory for matrix L\n");
        free(csrRowPtrL_tmp);
        free(csrColIdxL_tmp);
        free(csrValL_tmp);
        return -1;
    }
    
    int i,j,k;
    int tmp_col;
    VALUE_TYPE tmp_value;
    
    
    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;
    
    for (i = 0; i < m; i++)
    {
        for (j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
        {
            tmp_col=csrColIdxA[j];
            tmp_value=csrValA[j];
            for(k=j+1;k<csrRowPtrA[i+1];k++)
            {
                if(csrColIdxA[k]<tmp_col)
                {
                    csrColIdxA[j]=csrColIdxA[k];
                    csrValA[j]=csrValA[k];
                    csrColIdxA[k]=tmp_col;
                    csrValA[k]=tmp_value;
                    tmp_col=csrColIdxA[j];
                    tmp_value=csrValA[j];
                }
            }
            
            if (csrColIdxA[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = csrColIdxA[j];
                csrValL_tmp[nnz_pointer] = 1;//csrValA[j];
                nnz_pointer++;
            }
            else
            {
                break;
            }
        }
        
        csrColIdxL_tmp[nnz_pointer] = i;
        csrValL_tmp[nnz_pointer] = 1.0;
        nnz_pointer++;
        
        csrRowPtrL_tmp[i+1] = nnz_pointer;
    }
    
    nnzL = csrRowPtrL_tmp[m];
    
    // shrinking to the exact size is an optimization, so keep the larger
    // buffer if it fails
    int *csrColIdxL_shrunk = (int *)realloc(csrColIdxL_tmp, sizeof(int) * nnzL);
    if (csrColIdxL_shrunk != NULL)
        csrColIdxL_tmp = csrColIdxL_shrunk;

    VALUE_TYPE *csrValL_shrunk = (VALUE_TYPE *)realloc(csrValL_tmp, sizeof(VALUE_TYPE) * nnzL);
    if (csrValL_shrunk != NULL)
        csrValL_tmp = csrValL_shrunk;
    
    *nnzL_add=nnzL;
    *csrRowPtrL_tmp_add=csrRowPtrL_tmp;
    *csrColIdxL_tmp_add=csrColIdxL_tmp;
    *csrValL_tmp_add=csrValL_tmp;

    return 0;
}


void get_x_b(int m, int n, const int * csrRowPtrA, const int *csrColIdxA,
             const VALUE_TYPE *csrValA, VALUE_TYPE **x_add, VALUE_TYPE **b_add)
{
    VALUE_TYPE *x_ref = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n);
    int i,j;
    for ( i = 0; i < n; i++)
        x_ref[i] = 1;
    
    VALUE_TYPE *b = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m);
    
    // run spmv to get b
    for (i = 0; i < m; i++)
    {
        b[i] = 0;
        for (j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            b[i] += csrValA[j] * x_ref[csrColIdxA[j]];
    }
    *x_add=x_ref;
    *b_add=b;
}

#endif

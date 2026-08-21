#define KERNELS

void CanDeleteEdge(myInt *d_in_delete, myInt *isDecomposable,
                   const sycl::nd_item<1> &item, myInt &count,
                   myInt &contain_a, myInt &contain_b, myInt &which_ab) {
  myInt tid = item.get_local_id(0);
  int bid = item.get_group(0);
  myInt bdim = item.get_local_range(0);

  int n = *d_in_delete;
  myInt nCliques = *(d_in_delete + 1);
  myInt *CliquesDimens = d_in_delete + 2;
  myInt *Cliques = CliquesDimens + nCliques;
  // myInt  nTasks = *(Cliques+nCliques*n);
  myInt *d_a = Cliques + nCliques * n + 1;
  myInt *d_b = d_a + *(Cliques + nCliques * n); // nTasks;

  myInt i, j, k;
  int ii; // myInt n = *d_n;
  myInt a = d_a[bid];
  myInt b = d_b[bid];

  if (tid == 0) {
    count = 0;
  }

  SYNC;

  for (i = 0; i < nCliques; i++) {
    ii = i * n;
    if (tid == 0) {
      contain_a = 0;
      contain_b = 0;
    }
    SYNC;
    for (j = tid; j < CliquesDimens[i]; j += bdim) {
      k = Cliques[ii + j];
      if (k == a) {
        contain_a = 1;
      };
      if (k == b) {
        contain_b = 1;
      }
    }
    SYNC;
    if (tid == 0) {
      if (contain_a && contain_b) {
        count++;
        which_ab = i;
      }
    }
    // count is updated by thread 0 only, so it must be visible to every thread
    // before the loop-exit test below, or threads leave the loop divergently.
    SYNC;
    if (count > 1) {
      break;
    }
  }

  if (tid == 0) {
    if (count == 1) {
      isDecomposable[bid] = which_ab;
    } else {
      isDecomposable[bid] = -1;
    }
  }
}

// shared myInt demand: p+2*nTreeEdges+BLOCK_SIZE
void CanAddEdge(myInt *d_in_delete, myInt *d_in_add, myInt *isDecomposable,
                const sycl::nd_item<1> &item, myInt *shmem,
                myInt &nS, myInt &pR, myInt &pT, myInt &contain_a,
                myInt &contain_b, myInt &aSi, myInt &bSi, myInt &common_parent,
                myInt &a, myInt &b, myBool &flag) {
  myInt tid = item.get_local_id(0);
  int bid = item.get_group(0);
  myInt bdim = item.get_local_range(0);
  myInt *bi = shmem;
  int i, j, k, c, t;

  int n = (int)*d_in_delete;
  myInt nCliques = *(d_in_delete + 1);
  myInt *CliquesDimens = d_in_delete + 2;
  myInt *Cliques = CliquesDimens + nCliques;

  myInt *d_Labels = d_in_add;
  myInt nSeparators = *(d_in_add + n);
  myInt *SeparatorsDimens = d_in_add + n + 1;
  myInt *Separators = SeparatorsDimens + nSeparators;
  myInt nTreeEdges = *(Separators + n * nSeparators);
  myInt *TreeEdgeA = Separators + n * nSeparators + 1;
  myInt *TreeEdgeB = TreeEdgeA + nTreeEdges;
  myInt *d_Edge = TreeEdgeB + nTreeEdges;
  // myInt  nTasks = *(d_Edge + n*n);
  myInt *d_a = d_Edge + (n * n) + 1;
  myInt *d_b = d_a + *(d_Edge + (n * n)); // nTasks;

  myInt *R;
  myInt *T;
  myInt *S;
  myInt *contain_S;
  R = bi;
  bi += nTreeEdges;
  T = bi;
  bi += nTreeEdges;
  S = bi;
  bi += n;
  contain_S = bi; // bi += BLOCK_SIZE;

  if (tid == 0) {
    flag = 0;
    aSi = -1;
    bSi = -1;
    common_parent = 0;
    a = d_a[bid];
    b = d_b[bid];
    isDecomposable[bid] = 0;

    if (d_Labels[a] != d_Labels[b]) {
      flag = 1;
      isDecomposable[bid] = 1;
    }
  };
  SYNC;

  if (flag) {
    return;
  } // else ..

  if (tid == 0) {
    nS = 0;
    for (j = 0; j < n; j++) {
      if (d_Edge[a * n + j] && d_Edge[b * n + j]) {
        S[nS] = j;
        nS++;
      }
    };
    if (nS == 0) {
      flag = 1;
    }
  };
  SYNC;

  if (flag) {
    return;
  } // else ..

  // find higest-indexed cliques containing (a & S) and (b & S)
  for (i = 0; i < nCliques; i++) {
    if (tid == 0) {
      contain_a = 0;
      contain_b = 0;
    };
    contain_S[tid] = 0;
    SYNC;
    t = i * n;
    for (j = tid; j < CliquesDimens[i]; j += bdim) {
      c = Cliques[t + j];
      if (c == a) {
        contain_a = 1;
      };
      if (c == b) {
        contain_b = 1;
      }
      for (k = 0; k < nS; k++) {
        if (c == S[k]) {
          contain_S[tid]++;
        }
      }
    }
    // every thread contributes to contain_S[] and contain_a/contain_b above,
    // so those writes must be visible before thread 0 reduces them.
    SYNC;
    if (tid == 0) {
      k = 0;
      for (j = 0; j < BLOCK_SIZE; j++) {
        k += contain_S[j];
      }
      if (contain_a && (k == nS)) {
        aSi = i;
      };
      if (contain_b && (k == nS)) {
        bSi = i;
      }
    }
    SYNC;
  }

  if (tid == 0) { // find the path from aSi to root
    R[0] = -1;
    pR = -1;
    for (i = 0; i < nTreeEdges; i++) {
      if (TreeEdgeA[i] == aSi) {
        R[0] = i;
        pR = 0;
        break;
      }
    }
    for (i = R[0]; i >= 0; i--) {
      if (TreeEdgeA[i] == TreeEdgeB[R[pR]]) {
        pR++;
        R[pR] = i;
      }
    }
  } else if (tid == 1) {
    // find the path from bSi to root
    T[0] = -1;
    pT = -1;
    for (i = 0; i < nTreeEdges; i++) {
      if (TreeEdgeA[i] == bSi) {
        T[0] = i;
        pT = 0;
        break;
      }
    }
    for (i = T[0]; i >= 0; i--) {
      if (TreeEdgeA[i] == TreeEdgeB[T[pT]]) {
        pT++;
        T[pT] = i;
      }
    }
  }
  // R/pR are produced by thread 0 and T/pT by thread 1; both are read below.
  SYNC;

  if (tid == 0) {
    // find the branching point
    t = ((pR <= pT) ? pR : pT);
    for (i = 0; i <= t; i++) {
      if (TreeEdgeB[R[pR - i]] == TreeEdgeB[T[pT - i]]) {
        common_parent = i;
      } else {
        break;
      }
    }
    if (t != -1) {
      if (TreeEdgeA[R[pR - common_parent]] ==
          TreeEdgeA[T[pT - common_parent]]) {
        common_parent++;
      }
    }
  }
  SYNC;

  // check if S is in the path from R[0] to common_parent
  for (i = tid; i <= ((pR - common_parent) + (pT - common_parent) + 1);
       i += bdim) {
    if (i <= (pR - common_parent)) {
      if (SeparatorsDimens[R[i]] == nS) {
        contain_S[tid] = 0;
        t = R[i] * n;
        for (j = 0; j < nS; j++) {
          for (k = 0; k < nS; k++) {
            if (Separators[t + j] == S[k]) {
              contain_S[tid]++;
            }
          }
        }
        if (contain_S[tid] == nS) {
          flag = 1;
          isDecomposable[bid] = 1;
        }
      }
    } else {
      c = i - (pR - common_parent) - 1;
      if (SeparatorsDimens[T[c]] == nS) {
        contain_S[tid] = 0;
        t = T[c] * n;
        for (j = 0; j < nS; j++) {
          for (k = 0; k < nS; k++) {
            if (Separators[t + j] == S[k]) {
              contain_S[tid]++;
            }
          }
        }
        if (contain_S[tid] == nS) {
          flag = 1;
          isDecomposable[bid] = 1;
        }
      }
    }
  }
  SYNC;
}

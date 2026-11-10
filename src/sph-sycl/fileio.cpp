////////////////////////////////////////////////
// File input/output functions
////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include "sph.h"

static int fileNum = 0;

// Write fluid particle data to file
void writeFile(fluid_particle *particles, param *params)
{
    fluid_particle *p;
    FILE *fp ;
    int i;
    char name[64];
    //char* user;
    sprintf(name, "sim-%d.csv", fileNum);
    fp = fopen (name,"w");
    if (!fp) {
        printf("ERROR: error opening file %s\n",name);
        exit(1);
    }
    for(i=0; i<params->number_fluid_particles; i++) {
        p = &particles[i];
        fprintf(fp,"%f,%f,%f\n",p->pos.x,p->pos.y,p->pos.z);
    }
    fclose(fp);
    fileNum++;
    printf("wrote file: %s\n", name);
}

// Write boundary particle data to file
void writeBoundaryFile(boundary_particle *boundary, param *params)
{
    boundary_particle *k;
    FILE *fp ;
    int i;
    char name[64];
    sprintf(name, "boundary-%d.csv", fileNum);
    fp = fopen ( name,"w" );
    if (!fp) {
        printf("ERROR: error opening file %s\n",name);
        exit(1);
    }
    for(i=0; i<params->number_boundary_particles; i++) {
        k = &boundary[i];
        fprintf(fp,"%f,%f,%f\n",k->pos.x,k->pos.y,k->pos.z);
    }
    fclose(fp);
    printf("wrote file: %s\n", name);
}

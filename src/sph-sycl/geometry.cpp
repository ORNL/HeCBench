///////////////////////////////////////////////////////////////
//  Functions related to problem geometry
///////////////////////////////////////////////////////////////


#include <stdio.h>
#include <cmath>
#include "sph.h"

///////////////////////////////////////////////////////////////
// Construct the particle boundary box
// Setting particle normals require the explicit construction
///////////////////////////////////////////////////////////////
void constructBoundaryBox(boundary_particle *boundary_particles, AABB* boundary, param *params)
{
    double spacing = params->spacing_particle;
    
    // Create boundary particles with spacing h
    int num_x = std::ceil((boundary->max_x - boundary->min_x)/spacing);
    int num_y = std::ceil((boundary->max_y - boundary->min_y)/spacing);
    int num_z = std::ceil((boundary->max_z - boundary->min_z)/spacing);
    double min_x = boundary->min_x;
    double min_y = boundary->min_y;
    double min_z = boundary->min_z;
    double max_x = min_x + (num_x-1)*spacing;
    double max_y = min_y + (num_y-1)*spacing;
    double max_z = min_z + (num_z-1)*spacing;
    boundary->max_x = max_x;
    boundary->max_y = max_y;
    boundary->max_z = max_z;
    
    int i,nx,ny,nz;
    
    double recip_root_three = 1.0/std::sqrt(3.0);
    double recip_root_two   = 1.0/std::sqrt(2.0);
    i = 0;
    // Corner front bottom left
    boundary_particles[i].pos.x = min_x;
    boundary_particles[i].pos.y = max_y;
    boundary_particles[i].pos.z = min_z;
    boundary_particles[i].n.x = recip_root_three;
    boundary_particles[i].n.y = -recip_root_three;
    boundary_particles[i].n.z = recip_root_three;
    i++;
    // Corner front bottom right
    boundary_particles[i].pos.x = max_x;
    boundary_particles[i].pos.y = max_y;
    boundary_particles[i].pos.z = min_z;
    boundary_particles[i].n.x = -recip_root_three;
    boundary_particles[i].n.y = -recip_root_three;
    boundary_particles[i].n.z = recip_root_three;
    i++;
    // Corner front top left
    boundary_particles[i].pos.x = min_x;
    boundary_particles[i].pos.y = max_y;
    boundary_particles[i].pos.z = max_z;
    boundary_particles[i].n.x = recip_root_three;
    boundary_particles[i].n.y = -recip_root_three;
    boundary_particles[i].n.z = -recip_root_three;
    i++;
    // Corner front top right
    boundary_particles[i].pos.x = max_x;
    boundary_particles[i].pos.y = max_y;
    boundary_particles[i].pos.z = max_z;
    boundary_particles[i].n.x = -recip_root_three;
    boundary_particles[i].n.y = -recip_root_three;
    boundary_particles[i].n.z = -recip_root_three;
    i++;
    
    // Corner back bottom left
    boundary_particles[i].pos.x = min_x;
    boundary_particles[i].pos.y = min_y;
    boundary_particles[i].pos.z = min_z;
    boundary_particles[i].n.x = recip_root_three;
    boundary_particles[i].n.y = recip_root_three;
    boundary_particles[i].n.z = recip_root_three;
    i++;
    // Corner back bottom right
    boundary_particles[i].pos.x = max_x;
    boundary_particles[i].pos.y = min_y;
    boundary_particles[i].pos.z = min_z;
    boundary_particles[i].n.x = -recip_root_three;
    boundary_particles[i].n.y = recip_root_three;
    boundary_particles[i].n.z = recip_root_three;
    i++;
    // Corner back top left
    boundary_particles[i].pos.x = min_x;
    boundary_particles[i].pos.y = min_y;
    boundary_particles[i].pos.z = max_z;
    boundary_particles[i].n.x = recip_root_three;
    boundary_particles[i].n.y = recip_root_three;
    boundary_particles[i].n.z = -recip_root_three;
    i++;
    // Corner back top right
    boundary_particles[i].pos.x = max_x;
    boundary_particles[i].pos.y = min_y;
    boundary_particles[i].pos.z = max_z;
    boundary_particles[i].n.x = -recip_root_three;
    boundary_particles[i].n.y = recip_root_three;
    boundary_particles[i].n.z = -recip_root_three;
    i++;
    
    for (nx=0; nx<num_x-2; nx++) {
        // Bottom right row
        boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
        boundary_particles[i].pos.y = max_y;
        boundary_particles[i].pos.z = min_z;
        boundary_particles[i].n.x = 0.0;
        boundary_particles[i].n.y = -recip_root_two;
        boundary_particles[i].n.z = recip_root_two;
        i++;
        // Top right row
        boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
        boundary_particles[i].pos.y = max_y;
        boundary_particles[i].pos.z = max_z;
        boundary_particles[i].n.x = 0.0;
        boundary_particles[i].n.y = -recip_root_two;
        boundary_particles[i].n.z = -recip_root_two;
        i++;
        // Bottom left row
        boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
        boundary_particles[i].pos.y = min_y;
        boundary_particles[i].pos.z = min_z;
        boundary_particles[i].n.x = 0.0;
        boundary_particles[i].n.y = recip_root_two;
        boundary_particles[i].n.z = recip_root_two;
        i++;
        // Top left row
        boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
        boundary_particles[i].pos.y = min_y;
        boundary_particles[i].pos.z = max_z;
        boundary_particles[i].n.x = 0.0;
        boundary_particles[i].n.y = recip_root_two;
        boundary_particles[i].n.z = -recip_root_two;
        i++;
    }
    for (ny=0; ny<num_y-2; ny++) {
        // Bottom front row
        boundary_particles[i].pos.x = max_x;
        boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
        boundary_particles[i].pos.z = min_z;
        boundary_particles[i].n.x = -recip_root_two;
        boundary_particles[i].n.y = 0.0;
        boundary_particles[i].n.z = recip_root_two;
        i++;
        // Top front row
        boundary_particles[i].pos.x = max_x;
        boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
        boundary_particles[i].pos.z = max_z;
        boundary_particles[i].n.x = -recip_root_two;
        boundary_particles[i].n.y = 0.0;
        boundary_particles[i].n.z = -recip_root_two;
        i++;
        // Bottom back row
        boundary_particles[i].pos.x = min_x;
        boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
        boundary_particles[i].pos.z = min_z;
        boundary_particles[i].n.x = recip_root_two;
        boundary_particles[i].n.y = 0.0;
        boundary_particles[i].n.z = recip_root_two;
        i++;
        // Top back row
        boundary_particles[i].pos.x = min_x;
        boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
        boundary_particles[i].pos.z = max_z;
        boundary_particles[i].n.x = recip_root_two;
        boundary_particles[i].n.y = 0.0;
        boundary_particles[i].n.z = -recip_root_two;
        i++;
        
        for (nx=0; nx<num_x-2; nx++) {
            // Top face
            boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
            boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
            boundary_particles[i].pos.z = max_z;
            boundary_particles[i].n.x = 0.0;
            boundary_particles[i].n.y = 0.0;
            boundary_particles[i].n.z = -1.0;
            i++;
            // Bottom face
            boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
            boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
            boundary_particles[i].pos.z = min_z;
            boundary_particles[i].n.x = 0.0;
            boundary_particles[i].n.y = 0.0;
            boundary_particles[i].n.z = 1.0;
            i++;
        }
    }
    for (nz=0; nz<num_z-2; nz++) {
        // left front column
        boundary_particles[i].pos.x = min_x;
        boundary_particles[i].pos.y = max_y;
        boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
        boundary_particles[i].n.x = recip_root_two;
        boundary_particles[i].n.y = -recip_root_two;
        boundary_particles[i].n.z = 0.0;
        i++;
        // right front column
        boundary_particles[i].pos.x = max_x;
        boundary_particles[i].pos.y = max_y;
        boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
        boundary_particles[i].n.x = -recip_root_two;
        boundary_particles[i].n.y = -recip_root_two;
        boundary_particles[i].n.z = 0.0;
        i++;
        // left back column
        boundary_particles[i].pos.x = min_x;
        boundary_particles[i].pos.y = min_y;
        boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
        boundary_particles[i].n.x = recip_root_two;
        boundary_particles[i].n.y = recip_root_two;
        boundary_particles[i].n.z = 0.0;
        i++;
        // right back column
        boundary_particles[i].pos.x = max_x;
        boundary_particles[i].pos.y = min_y;
        boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
        boundary_particles[i].n.x = -recip_root_two;
        boundary_particles[i].n.y = recip_root_two;
        boundary_particles[i].n.z = 0.0;
        i++;
        for (nx=0; nx<num_x-2; nx++) {
            // Front face
            boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
            boundary_particles[i].pos.y = max_y;
            boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
            boundary_particles[i].n.x = 0.0;
            boundary_particles[i].n.y = -1.0;
            boundary_particles[i].n.z = 0.0;
            i++;
            // Back face
            boundary_particles[i].pos.x = min_x + spacing + nx*spacing;
            boundary_particles[i].pos.y = min_y;
            boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
            boundary_particles[i].n.x = 0.0;
            boundary_particles[i].n.y = 1.0;
            boundary_particles[i].n.z = 0.0;
            i++;
        }
        for (ny=0; ny<num_y-2; ny++) {
            // Left face
            boundary_particles[i].pos.x = min_x;
            boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
            boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
            boundary_particles[i].n.x = 1.0;
            boundary_particles[i].n.y = 0.0;
            boundary_particles[i].n.z = 0.0;
            i++;
            // Right face
            boundary_particles[i].pos.x = max_x;
            boundary_particles[i].pos.y = min_y + spacing + ny*spacing;
            boundary_particles[i].pos.z = min_z + spacing + nz*spacing;
            boundary_particles[i].n.x = -1.0;
            boundary_particles[i].n.y = 0.0;
            boundary_particles[i].n.z = 0.0;
            i++;
        }
    }
    params->number_boundary_particles = i;
    params->number_particles = params->number_fluid_particles + params->number_boundary_particles;
    
}

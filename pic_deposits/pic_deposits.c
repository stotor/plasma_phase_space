#include <math.h>

int mod(int a, int b)
{
  return (a%b + b) % b;
}

int field_index(int y, int x, int n_y, int n_x)
{
  int i, j;
  i = mod(x, n_x);
  j = mod(y, n_y);
  
  return j * n_x + i;
}

void deposit_particle_2d_0(double x, double y, double *field, double charge,
			   int n_x, int n_y, double cell_width)
{
  int i_ll, j_ll, i_ngp, j_ngp;
  x = x / cell_width - 0.5;
  y = y / cell_width - 0.5;
    
  i_ll = floor(x);
  j_ll = floor(y);
    
  if ((x-i_ll)<0.5) {
    i_ngp = i_ll;
  }
  else {
    i_ngp = i_ll+1;
  }
  
  if ((y-j_ll)<0.5) {
    j_ngp = j_ll;
  }
  else {
    j_ngp = j_ll+1;
  }
  
  field[field_index(j_ngp, i_ngp, n_y, n_x)] += charge;
    
  return;
}


void deposit_particle_2d_1(double x, double y, double *field, double charge,
			   int n_x, int n_y, double cell_width)
{
  int i, j, i_ll, j_ll;
  double dx, dy, wx[2], wy[2];
  
  x = x / cell_width - 0.5;
  y = y / cell_width - 0.5;
    
  i_ll = floor(x);
  j_ll = floor(y);

  dx = x - (i_ll + 0.5);
  dy = y - (j_ll + 0.5);

  wx[0] = 0.5 - dx;
  wx[1] = 0.5 + dx;

  wy[0] = 0.5 - dy;
  wy[1] = 0.5 + dy;

  for (j=0; j<2; j++) {
    for (i=0; i<2; i++) {
      field[field_index(j_ll+j, i_ll+i, n_y, n_x)] += charge * wx[i] * wy[j];
    }
  }
    
  return;
}

void deposit_particle_2d_2(double x, double y, double *field, double charge,
			   int n_x, int n_y, double cell_width)
{
  int i, j, i_ll, j_ll, i_ngp, j_ngp;
  double dx, dy, wx[3], wy[3];
  
  x = x / cell_width - 0.5;
  y = y / cell_width - 0.5;
    
  i_ll = floor(x);
  j_ll = floor(y);

  if ((x-i_ll)<0.5) {
    i_ngp = i_ll;
  } else {
    i_ngp = i_ll+1;
  }
        
  if ((y-j_ll)<0.5) {
    j_ngp = j_ll;
  } else {
    j_ngp = j_ll+1;
  }

  dx = x - i_ngp;
  dy = y - j_ngp;

  wx[0] = pow((1.0 - 2.0 * dx), 2) / 8.0;
  wx[1] = 0.75 - dx*dx;
  wx[2] = pow((1.0 + 2.0 * dx), 2) / 8.0;
    
  wy[0] = pow((1.0 - 2.0 * dy), 2) / 8.0;
  wy[1] = 0.75 - dy * dy;
  wy[2] = pow((1.0 + 2.0 * dy), 2) / 8.0;

  for (j=0; j<3; j++) {
    for (i=0; i<3; i++) {
      field[field_index(j_ngp-1+j, i_ngp-1+i, n_y, n_x)] += charge * wx[i] * wy[j];
    }
  }
    
  return;
}

void deposit_particle_2d_3(double x, double y, double *field, double charge,
			   int n_x, int n_y, double cell_width)
{
  int i, j, i_ll, j_ll;
  double dx, dy, wx[4], wy[4];
  
  x = x / cell_width - 0.5;
  y = y / cell_width - 0.5;
    
  i_ll = floor(x);
  j_ll = floor(y);

  dx = x - (i_ll + 0.5);
  dy = y - (j_ll + 0.5);
  
  wx[0] = -1.0 * pow((-0.5 + dx), 3) / 6.0;
  wx[1] = (4.0 - 6.0 * pow((0.5 + dx), 2) + 3.0 * pow((0.5 + dx), 3)) / 6.0;
  wx[2] = (23.0 + 30.0 * dx - 12.0 * dx * dx - 24.0 * dx * dx * dx) / 48.0;
  wx[3] = pow((0.5 + dx), 3) / 6.0;

  wy[0] = -1.0 * pow((-0.5 + dy), 3) / 6.0;
  wy[1] = (4.0 - 6.0 * pow((0.5 + dy), 2) + 3.0 * pow((0.5 + dy), 3)) / 6.0;
  wy[2] = (23.0 + 30.0 * dy - 12.0 * dy * dy - 24.0 * dy * dy * dy) / 48.0;
  wy[3] = pow((0.5 + dy), 3) / 6.0;

  for (j=0; j<4; j++) {
    for (i=0; i<4; i++) {
      field[field_index(j_ll-1+j, i_ll-1+i, n_y, n_x)] += charge * wx[i] * wy[j];
    }
  }
    
  return;
}

void deposit_particle_2d_4(double x, double y, double *field, double charge,
			   int n_x, int n_y, double cell_width)
{
  int i, j, i_ll, j_ll, i_ngp, j_ngp;
  double dx, dy, wx[5], wy[5];
  
  x = x / cell_width - 0.5;
  y = y / cell_width - 0.5;
    
  i_ll = floor(x);
  j_ll = floor(y);

  if ((x-i_ll)<0.5) {
    i_ngp = i_ll;
  } else {
    i_ngp = i_ll+1;
  }
        
  if ((y-j_ll)<0.5) {
    j_ngp = j_ll;
  } else {
    j_ngp = j_ll+1;
  }

  dx = x - i_ngp;
  dy = y - j_ngp;

  wx[0] = pow((1.0 - 2.0 * dx), 4) / 384.0;
  wx[1] = (19.0 - 44.0 * dx + 24.0 * dx * dx + 16.0 * pow(dx, 3) - 16.0 * pow(dx, 4)) / 96.0;
  wx[2] = 0.5989583333333334 - (5.0 * dx * dx) / 8.0 + pow(dx, 4) / 4.0;
  wx[3] = (19.0 + 44.0 * dx + 24.0 * dx * dx - 16.0 * pow(dx, 3) - 16.0 * pow(dx, 4)) / 96.0;
  wx[4] = pow((1.0 + 2.0 * dx), 4) / 384.0;

  wy[0] = pow((1.0 - 2.0 * dy), 4) / 384.0;
  wy[1] = (19.0 - 44.0 * dy + 24.0 * dy * dy + 16.0 * pow(dy, 3) - 16.0 * pow(dy, 4)) / 96.0;
  wy[2] = 0.5989583333333334 - (5.0 * dy * dy) / 8.0 + pow(dy, 4) / 4.0;
  wy[3] = (19.0 + 44.0 * dy + 24.0 * dy * dy - 16.0 * pow(dy, 3) - 16.0 * pow(dy, 4)) / 96.0;
  wy[4] = pow((1.0 + 2.0 * dy), 4) / 384.0;


  for (j=0; j<5; j++) {
    for (i=0; i<5; i++) {
      field[field_index(j_ngp-2+j, i_ngp-2+i, n_y, n_x)] += charge * wx[i] * wy[j];
    }
  }
    
  return;
}

void deposit_species(double *particle_positions, double *field,
		     double *particle_charges, int n_x, int n_y, int n_p,
		     double cell_width, int order)
{
  int i;
  switch (order)
    {
    case 0 :
      for (i=0; i<n_p; i++) {
	deposit_particle_2d_0(particle_positions[2*i+1], particle_positions[2*i],
			      field, particle_charges[i], n_x, n_y, cell_width);
      }
      break;
      
    case 1 :
      for (i=0; i<n_p; i++) {
	deposit_particle_2d_1(particle_positions[2*i+1], particle_positions[2*i],
			      field, particle_charges[i], n_x, n_y, cell_width);
      }
      break;
      
    case 2 :
      for (i=0; i<n_p; i++) {
	deposit_particle_2d_2(particle_positions[2*i+1], particle_positions[2*i],
			      field, particle_charges[i], n_x, n_y, cell_width);
      }
      break;
      
    case 3 :
      for (i=0; i<n_p; i++) {
	deposit_particle_2d_3(particle_positions[2*i+1], particle_positions[2*i],
			      field, particle_charges[i], n_x, n_y, cell_width);
      }
      
      break;
    case 4 :
      for (i=0; i<n_p; i++) {
	deposit_particle_2d_4(particle_positions[2*i+1], particle_positions[2*i],
			      field, particle_charges[i], n_x, n_y, cell_width);
      }
      break;
    }
  
  return;
}

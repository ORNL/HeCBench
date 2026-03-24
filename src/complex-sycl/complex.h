#if !defined(COMPLEX_H_)
#define COMPLEX_H_

#include <cmath>       /* import fabs, sqrt */

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

typedef sycl::float2 FloatComplex;

static inline float Crealf (FloatComplex x) 
{ 
    return x.x(); 
}

static inline float Cimagf (FloatComplex x) 
{ 
    return x.y(); 
}

static inline FloatComplex make_FloatComplex (float r, float i)
{
    FloatComplex res;
    res.x() = r;
    res.y() = i;
    return res;
}

static inline FloatComplex Conjf (FloatComplex x)
{
    return make_FloatComplex (Crealf(x), -Cimagf(x));
}

static inline FloatComplex Caddf (FloatComplex x, FloatComplex y)
{
    return make_FloatComplex (Crealf(x) + Crealf(y), 
                              Cimagf(x) + Cimagf(y));
}

static inline FloatComplex Csubf (FloatComplex x, FloatComplex y)
{
    return make_FloatComplex (Crealf(x) - Crealf(y), 
                              Cimagf(x) - Cimagf(y));
}

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we 
 * don't do it either to stay competitive.
 */
static inline FloatComplex Cmulf (FloatComplex x, FloatComplex y)
{
    FloatComplex prod;
    prod = make_FloatComplex  ((Crealf(x) * Crealf(y)) - 
                               (Cimagf(x) * Cimagf(y)),
                               (Crealf(x) * Cimagf(y)) + 
                               (Cimagf(x) * Crealf(y)));
    return prod;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
static inline FloatComplex Cdivf (FloatComplex x, FloatComplex y)
{
    FloatComplex quot;
    float s = sycl::fabs(Crealf(y)) + sycl::fabs(Cimagf(y));
    float oos = 1.0f / s;
    float ars = Crealf(x) * oos;
    float ais = Cimagf(x) * oos;
    float brs = Crealf(y) * oos;
    float bis = Cimagf(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    quot = make_FloatComplex (((ars * brs) + (ais * bis)) * oos,
                              ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

/* 
 * We would like to call hypotf(), but it's not available on all platforms.
 * This discrete implementation guards against intermediate underflow and 
 * overflow by scaling. Otherwise we would lose half the exponent range. 
 * There are various ways of doing guarded computation. For now chose the 
 * simplest and fastest solution, however this may suffer from inacracies 
 * if sqrt and division are not IEEE compliant. 
 */
static inline float Cabsf (FloatComplex x)
{
    float a = Crealf(x);
    float b = Cimagf(x);
    float v, w, t;
    a = sycl::fabs(a);
    b = sycl::fabs(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0f + t * t;
    t = v * sycl::sqrt(t);
    if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
        t = v + w;
    }
    return t;
}

/* Double precision */
typedef sycl::double2 DoubleComplex;

static inline double Creal (DoubleComplex x) 
{ 
    return x.x(); 
}

static inline double Cimag (DoubleComplex x) 
{ 
    return x.y(); 
}

static inline DoubleComplex make_DoubleComplex (double r, double i)
{
    DoubleComplex res;
    res.x() = r;
    res.y() = i;
    return res;
}

static inline DoubleComplex Conj(DoubleComplex x)
{
    return make_DoubleComplex (Creal(x), -Cimag(x));
}

static inline DoubleComplex Cadd(DoubleComplex x, DoubleComplex y)
{
    return make_DoubleComplex (Creal(x) + Creal(y), 
                               Cimag(x) + Cimag(y));
}

static inline DoubleComplex Csub(DoubleComplex x, DoubleComplex y)
{
    return make_DoubleComplex (Creal(x) - Creal(y), 
                               Cimag(x) - Cimag(y));
}

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we 
 * don't do it either to stay competitive.
 */
static inline DoubleComplex Cmul(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex prod;
    prod = make_DoubleComplex ((Creal(x) * Creal(y)) - 
                               (Cimag(x) * Cimag(y)),
                               (Creal(x) * Cimag(y)) + 
                               (Cimag(x) * Creal(y)));
    return prod;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
static inline DoubleComplex Cdiv(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex quot;
    double s = (sycl::fabs(Creal(y))) + (sycl::fabs(Cimag(y)));
    double oos = 1.0 / s;
    double ars = Creal(x) * oos;
    double ais = Cimag(x) * oos;
    double brs = Creal(y) * oos;
    double bis = Cimag(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    quot = make_DoubleComplex (((ars * brs) + (ais * bis)) * oos,
                                 ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Otherwise we would lose half the exponent range. There are
 * various ways of doing guarded computation. For now chose the simplest
 * and fastest solution, however this may suffer from inacracies if sqrt
 * and division are not IEEE compliant.
 */
static inline double Cabs (DoubleComplex x)
{
    double a = Creal(x);
    double b = Cimag(x);
    double v, w, t;
    a = sycl::fabs(a);
    b = sycl::fabs(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0 + t * t;
    t = v * sycl::sqrt(t);
    if ((v == 0.0) || 
        (v > 1.79769313486231570e+308) || (w > 1.79769313486231570e+308)) {
        t = v + w;
    }
    return t;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

/* aliases */
typedef FloatComplex Complex;
static inline Complex make_Complex (float x, float y) 
{ 
    return make_FloatComplex (x, y); 
}

/* float-to-double promotion */
static inline DoubleComplex ComplexFloatToDouble (FloatComplex c)
{
    return make_DoubleComplex ((double)Crealf(c), (double)Cimagf(c));
}

static inline FloatComplex ComplexDoubleToFloat (DoubleComplex c)
{
    return make_FloatComplex ((float)Creal(c), (float)Cimag(c));
}

static inline  Complex Cfmaf( Complex x, Complex y, Complex d)
{
    float real_res;
    float imag_res;
    
    real_res = (Crealf(x) *  Crealf(y)) + Crealf(d);
    imag_res = (Crealf(x) *  Cimagf(y)) + Cimagf(d);
            
    real_res = -(Cimagf(x) * Cimagf(y))  + real_res;  
    imag_res =  (Cimagf(x) *  Crealf(y)) + imag_res;          
     
    return make_Complex(real_res, imag_res);
}

static inline  DoubleComplex Cfma( DoubleComplex x, DoubleComplex y, DoubleComplex d)
{
    double real_res;
    double imag_res;
    
    real_res = (Creal(x) *  Creal(y)) + Creal(d);
    imag_res = (Creal(x) *  Cimag(y)) + Cimag(d);
            
    real_res = -(Cimag(x) * Cimag(y))  + real_res;  
    imag_res =  (Cimag(x) *  Creal(y)) + imag_res;     
     
    return make_DoubleComplex(real_res, imag_res);
}

#endif /* !defined(COMPLEX_H_) */

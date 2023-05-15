#include "iir_coeff.h"

using namespace Halide;

using std::vector;
using std::complex;

/** Compute the factorial of an integer */
static inline int factorial(int k)
{
	assert(k >= 0);
	int r = 1;
	for (int i = 1; i <= k; i++)
	{
		r *= i;
	}
	return r;
}

/** Compute the i-th binomial coeff of the expansion of (1-r*x)^n */
static inline float binomial_coeff(int n, int i, float r)
{
	int n_choose_i = factorial(n) / (factorial(i) * factorial(n - i));
	return (pow(-r, i) * float(n_choose_i));
}

/**
 *  @brief Compute recursive filtering scaling factor
 *
 *  Compute the scaling factor of the recursive filter representing a
 *  true Gaussian filter convolution with arbitrary support sigma.
 *
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Scaling factor q of the recursive filter approximation
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
static float qs(const float& s)
{
	return 0.00399341 + 0.4715161 * s;
}


/**
 *  @brief Rescale poles of the recursive filtering z-transform
 *
 *  Given a complex-valued pole on |z|=1 ROC of the recursive filter
 *  z-transform compute a rescaled pole representing a true Gaussian
 *  filter convolution with arbitrary support sigma.
 *
 *  @param[in] d Complex-valued pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled complex-valued pole of the recursive filter approximation
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
static std::complex<double> ds(const std::complex<double>& d, const float& s)
{
	double q = qs(s);
	return std::polar(std::pow(std::abs(d), 1.0 / q), std::arg(d) / q);
}

/**
 *  @brief Rescale poles in the real-axis of the recursive filtering z-transform
 *
 *  Given a real pole on |z|=1 ROC of the recursive filter z-transform
 *  compute a rescaled pole representing a true Gaussian filter
 *  convolution with arbitrary support sigma.
 *
 *  @param[in] d Real pole of a stable recursive filter
 *  @param[in] s Sigma support of the true Gaussian filter
 *  @return Rescaled real pole of the recursive filter approximation
 *  @tparam float Sigma value type
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
static float ds(const float& d, const float& s)
{
	return std::pow(d, 1.0 / qs(s));
}

/**
 *  @brief Compute first-order weights
 *
 *  Given a Gaussian sigma value compute the feedforward and feedback
 *  first-order coefficients.
 *
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
static void weights1(const float& s, float& b0, float& a1)
{
	const float d3 = 1.86543;
	float d = ds(d3, s);
	b0 = -(1.0 - d) / d;
	a1 = -1.0 / d;
}

/**
 *  @brief Compute first and second-order weights
 *
 *  Given a Gaussian sigma value compute the feedforward and feedback
 *  first- and second-order coefficients.
 *
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *  @param[out] a2 Feedback second-order coefficient
 *
 *  Code taken from "GPU efficient recursive filtering and summed area tables"
 *  [Nehab et al. 2011]
 *
 *  See "Recursive Gaussian derivative filters" [van Vliet et al. 1998]
 *  for details on derivation.
 */
static void weights2(const float& s, float& b0, float& a1, float& a2)
{
	const std::complex<double> d1(1.41650, 1.00829);
	std::complex<double> d = ds(d1, s);
	float n2 = std::abs(d);
	n2 *= n2;
	float re = std::real(d);
	b0 = (1.0 - 2.0 * re + n2) / n2;
	a1 = -2.0 * re / n2;
	a2 = 1.0 / n2;
}

/**
 *  @brief Compute third order recursive filter weights for approximating Gaussian
 *
 *  @param[in] s Gaussian sigma
 *  @param[out] b0 Feedforward coefficient
 *  @param[out] a1 Feedback first-order coefficient
 *  @param[out] a2 Feedback second-order coefficient
 *  @param[out] a3 Feedback third-order coefficient
 *
 *  Coefficients equivalent to applying a first order recursive filter followed
 *  by second order filter
 */
static void weights3(const float& s, float& b0, float& a1, float& a2, float& a3)
{
	float b10, b20;
	float a11, a21, a22;
	weights1(s, b10, a11);
	weights2(s, b20, a21, a22);
	a1 = a11 + a21;
	a2 = a11 * a21 + a22;
	a3 = a11 * a22;
	b0 = b10 * b20;
}


vector<float> gaussian_weights(float sigma, int order)
{
	float b0 = 0.0;
	vector<float> a(order + 1, 0.0);

	switch (order)
	{
	case 1: weights1(sigma, a[0], a[1]); break;
	case 2: weights2(sigma, a[0], a[1], a[2]); break;
	default:weights3(sigma, a[0], a[1], a[2], a[3]); break;
	}

	for (int i = 1; i < a.size(); i++)
	{
		a[i] = -a[i];
	}

	return a;
}

Expr gaussian(Expr x, float mu, float sigma)
{
	Expr xx = Internal::Cast::make(type_of<float>(), x);
	Expr y = (xx - mu) / sigma;
	return Halide::fast_exp(-0.5f * y * y) / (sigma * 2.50662827463f);
}
Expr gaussDerivative(Expr x, float mu, float sigma)
{
	Expr xx = Internal::Cast::make(type_of<float>(), x);
	Expr y = (xx - mu) / sigma;
	return (mu - xx) * Halide::fast_exp(-0.5f * y * y) / (sigma * sigma * sigma * 2.50662827463f);
}
Expr gaussIntegral(Expr x, float mu, float sigma)
{
	Expr xx = Internal::Cast::make(type_of<float>(), x);
	return 0.5f * (1.0f + Halide::erf((xx - mu) / (sigma * 1.41421356237f)));
}
float gaussian(float x, float mu, float sigma)
{
	float y = (x - mu) / sigma;
	return (std::exp(-0.5 * y * y) / (sigma * 2.50662827463));
}
float gaussDerivative(float x, float mu, float sigma)
{
	float y = (x - mu) / sigma;
	return ((mu - x) * std::exp(-0.5 * y * y) / (sigma * sigma * sigma * 2.50662827463));
}
float gaussIntegral(float x, float mu, float sigma)
{
	return (0.5 * (1.0 + erf((x - mu) / (sigma * 1.41421356237))));
}

int gaussian_box_filter(int k, float sigma)
{
	float sum = 0.0;
	float alpha = 0.005;
	int sum_limit = int(std::floor((float(k) - 1.0) / 2.0));
	for (int i = 0; i <= sum_limit; i++)
	{
		int f_k = factorial(k);
		int f_i = factorial(i);
		int f_k_i = factorial(k - i);
		int f_k_1 = factorial(k - 1);
		float f = float(f_k / (f_i * f_k_i));
		float p = std::pow(-1.0, i) / float(f_k_1);
		sum += p * f * std::pow((float(k) / 2.0 - i), k - 1);
	}
	sum = std::sqrt(2.0 * 3.141592) * (sum + alpha) * sigma;
	return int(std::ceil(sum));
}

vector<float> integral_image_coeff(int n)
{
	vector<float> coeff(n + 1, 0.0f);

	// set feedforward coeff = 1.0f
	coeff[0] = 1.0f;

	// feedback coeff are binomial expansion of (1-x)^n multiplied by -1
	for (int i = 1; i <= n; i++)
	{
		coeff[i] = -1.0f * binomial_coeff(n, i, 1.0f);
	}

	return coeff;
}

vector<float> overlap_feedback_coeff(vector<float> a, vector<float> b)
{
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = -a[i];
	}
	for (int i = 0; i < b.size(); i++)
	{
		b[i] = -b[i];
	}

	a.insert(a.begin(), 1.0f);
	b.insert(b.begin(), 1.0f);

	vector<float> c(a.size() + b.size() - 1, 0.0f);

	for (int i = 0; i < c.size(); i++)
	{
		for (int j = 0; j <= i; j++)
		{
			if (j < a.size() && i - j < b.size())
			{
				c[i] += a[j] * b[i - j];
			}
		}
	}

	c.erase(c.begin());
	for (int i = 0; i < c.size(); i++)
	{
		c[i] = -c[i];
	}

	return c;
}

void make_filter(vector<float>& result_b, vector<float>& result_a, const complex<double> alpha[], const complex<double> beta[], int K, double sigma)
{
	const double denom = sigma * M_SQRT2PI;
	complex<double> b[DERICHE_MAX_K], a[DERICHE_MAX_K + 1];

	b[0] = alpha[0];
	a[0] = complex<double>(1.0, 0.0);
	a[1] = beta[0];

	for (int k = 1; k < K; k++)
	{
		b[k] = beta[k] * b[k - 1];

		for (int j = k - 1; j > 0; --j)
			b[j] = b[j] + (beta[k] * b[j - 1]);

		for (int j = 0; j <= k; ++j)
			b[j] = b[j] + (alpha[k] * a[j]);

		a[k + 1] = beta[k] * a[k];

		for (int j = k; j > 0; --j)
			a[j] = a[j] + (beta[k] * a[j - 1]);
	}

	for (int k = 0; k < K; ++k)
	{
		result_b[k] = (float)(b[k].real() / denom);
		result_a[k] = -(float)a[k + 1].real();
	}

	return;
}

// rep[3] = {a, b_causal, b_anti_causal}
vector<vector<float>> gaussian_coeff_for_deriche(double sigma, int order)
{
	static const complex<double> alpha[DERICHE_MAX_K - DERICHE_MIN_K + 1][4]{
		{{0.48145, 0.971}, {0.48145, -0.971}},
		{{-0.44645, 0.5105}, {-0.44645, -0.5105}, {1.898, 0}},
		{{0.84, 1.8675}, {0.84, -1.8675},
			{-0.34015, -0.1299}, {-0.34015, 0.1299}}
	};
	static const complex<double> lambda[DERICHE_MAX_K - DERICHE_MIN_K + 1][4]{
		{{1.26, 0.8448}, {1.26, -0.8448}},
		{{1.512, 1.475}, {1.512, -1.475}, {1.556, 0}},
		{{1.783, 0.6318}, {1.783, -0.6318},
			{1.723, 1.997}, {1.723, -1.997}}
	};
	complex<double> beta[DERICHE_MAX_K];

	assert(sigma > 0 && DERICHE_VALID_K(order));

	for (int k = 0; k < order; k++)
	{
		double temp = std::exp(-lambda[order - DERICHE_MIN_K][k].real() / sigma);
		beta[k] = complex<double>(-temp * std::cos(lambda[order - DERICHE_MIN_K][k].imag() / sigma),
			temp * std::sin(lambda[order - DERICHE_MIN_K][k].imag() / sigma));
	}

	vector<float> a(order), b_causal(order), b_anticausal(order);

	make_filter(b_causal, a, alpha[order - DERICHE_MIN_K], beta, order, sigma);

	b_anticausal[0] = 0.0f;

	for (int k = 1; k < order; k++)
		b_anticausal[k] = b_causal[k] + a[k - 1] * b_causal[0];

	b_anticausal.erase(b_anticausal.begin());

	b_anticausal.push_back(a[order - 1] * b_causal[0]);

	vector<vector<float>> coeff;
	vector<float> temp = a;
	temp.insert(temp.begin(), b_causal.begin(), b_causal.end());
	coeff.push_back(temp);

	temp = a;
	temp.insert(temp.begin(), b_anticausal.begin(), b_anticausal.end());
	coeff.push_back(temp);

	return coeff;
}

static double variance(const complex<double>* poles0, int K, double q)
{
	complex<double> sum = { 0, 0 };
	int k;

	for (k = 0; k < K; ++k)
	{
		complex<double> z = pow(poles0[k], 1 / q), denom = z;
		denom -= 1;
		/* Compute sum += z / (z - 1)^2. */
		sum = sum + z / (denom * denom);
	}

	return 2 * sum.real();
}

/**
 * \brief Derivative of variance with respect to q
 * \param poles0    unscaled pole locations
 * \param q         rescaling parameter
 * \param K         number of poles
 * \return derivative of variance with respect to q
 * \ingroup vyv_gaussian
 *
 * This function is used by compute_q() in solving for q.
 */
static double dq_variance(const complex<double>* poles0, int K, double q)
{
	complex<double> sum = { 0, 0 };
	int k;

	for (k = 0; k < K; ++k)
	{
		complex<double> z = pow(poles0[k], 1 / q), w = z, denom = z;
		w += 1;
		denom -= 1;
		/* Compute sum += z log(z) (z + 1) / (z - 1)^3 */
		sum = sum + (z * log(z) * w) / pow(denom, 3);
	}

	return (2 / q) * sum.real();
}

/**
 * \brief Compute q for a desired sigma using Newton's method
 * \param poles0    unscaled pole locations
 * \param K         number of poles
 * \param sigma     the desired sigma
 * \param q0        initial estimate of q
 * \return refined value of q
 * \ingroup vyv_gaussian
 *
 * This routine uses Newton's method to solve for the value of q so that the
 * filter achieves the specified variance,
 * \f[ \operatorname{var}(h) = \sum_{k=1}^K \frac{2 d_k^{1/q}}
							   {(d_k^{1/q} - 1)^2} = \sigma^2, \f]
 * where the \f$ d_k \f$ are the unscaled pole locations.
 */
static double compute_q(const complex<double>* poles0, int K,
	double sigma, double q0)
{
	double sigma2 = sigma * sigma;
	double q = q0;
	int i;

	for (i = 0; i < YVY_NUM_NEWTON_ITERATIONS; ++i)
		q -= (variance(poles0, K, q) - sigma2)
		/ dq_variance(poles0, K, q);

	return q;
}

static void expand_pole_product(double* c, const complex<double>* poles, int K)
{
	complex<double> denom[VYV_MAX_K + 1];
	int k, j;

	assert(K <= VYV_MAX_K);
	denom[0] = poles[0];
	denom[1] = complex<double>(-1, 0);

	for (k = 1; k < K; ++k)
	{
		denom[k + 1] = -denom[k];

		for (j = k; j > 0; --j)
			denom[j] = denom[j] * poles[k] - denom[j - 1];

		denom[0] = denom[0] * poles[k];
	}

	for (k = 1; k <= K; ++k)
		c[k] = (denom[k] / denom[0]).real();

	for (c[0] = 1, k = 1; k <= K; ++k)
		c[0] += c[k];

	return;
}


vector<float> gaussian_coeff_for_vyv(float sigma, int order)
{
	/* Optimized unscaled pole locations. */
	static const complex<double> poles0[VYV_MAX_K - VYV_MIN_K + 1][5] =
	{
			{{1.4165, 1.00829}, {1.4165, -1.00829}, {1.86543, 0}},
			{{1.13228, 1.28114}, {1.13228, -1.28114},
				{1.78534, 0.46763}, {1.78534, -0.46763}},
			{{0.8643, 1.45389}, {0.8643, -1.45389},
				{1.61433, 0.83134}, {1.61433, -0.83134}, {1.87504, 0}}
	};
	complex<double> poles[VYV_MAX_K];
	double q, filter[VYV_MAX_K + 1];
	double A[VYV_MAX_K * VYV_MAX_K], inv_A[VYV_MAX_K * VYV_MAX_K];
	int i, j, matrix_size;

	assert(sigma > 0 && VYV_VALID_K(order));

	/* Make a crude initial estimate of q. */
	q = sigma / 2;
	/* Compute an accurate value of q using Newton's method. */
	q = compute_q(poles0[order - VYV_MIN_K], order, sigma, q);

	for (i = 0; i < order; ++i)
		poles[i] = pow(poles0[order - VYV_MIN_K][i], 1 / q);

	/* Compute the filter coefficients b_0, a_1, ..., a_K. */
	expand_pole_product(filter, poles, order);

	vector<float> a(order + 1, 0.0);

	a[0] = filter[0];
	for (int i = 1; i <= order; i++)
	{
		a[i] = -filter[i];
	}

	return a;
}
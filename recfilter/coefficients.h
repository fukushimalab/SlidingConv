#ifndef _COEFFICIENTS_H_
#define _COEFFICIENTS_H_

Halide::Buffer<float> matrix_B(
	Halide::Buffer<float> feedfwd_coeff,
	Halide::Buffer<float> feedback_coeff,
	int scan_id,
	int tile_width,
	bool clamp_border);

Halide::Buffer<float> matrix_R(
	Halide::Buffer<float> feedback_coeff,
	int scan_id,
	int tile_width);

Halide::Buffer<float> matrix_transpose(Halide::Buffer<float> A);
template<typename T> Halide::Buffer<T> matrix_mult(Halide::Buffer<T> A, Halide::Buffer<T> B)
{
	assert(A.width() == B.height());

	int num_rows = A.height();
	int num_cols = B.width();
	int num_common = A.width();

	Halide::Buffer<T> C(num_cols, num_rows);

	for (int i = 0; i < C.width(); i++)
	{
		for (int j = 0; j < C.height(); j++)
		{
			C(i, j) = 0.0;
		}
	}
	for (int i = 0; i < C.height(); i++)
	{
		for (int j = 0; j < C.width(); j++)
		{
			for (int k = 0; k < num_common; k++)
			{
				C(j, i) += A(k, i) * B(j, k);
			}
		}
	}
	return C;
};
Halide::Buffer<float> matrix_antidiagonal(int size);


#endif // _COEFFICIENTS_H_

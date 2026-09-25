#include "NN.hpp"
#include "examples/standard_mnist.hpp"
// #include "examples/inverted_mnist.hpp"
// #include "examples/model_export_n_import.hpp"
// #include "examples/import_model.hpp"
// #include "utils.hpp"

int main()
{
	Config config(
        30,
        32,
        9.0F,
        {784, 128, 30, 10},
        Threading::Single,
        0
    );

	standard_mnist(config);
	// inverted_mnist();
	// import_model("data/output/mnist_inverted_10-128-256-512-784_ep20_lr15p000.bin");
	// model_export_n_import(10, 32, 9.0F);

	return 0;
}

#include <algorithm>
#include "NN.hpp"
#include "data_loaders.hpp"

int main() {

	auto training_data = load_training_data("data/train_images.bin", "data/train_labels.bin", 50000);
	std::cout << "Training data loaded." << std::endl;
	
	auto test_data = load_test_data("data/test_images.bin", "data/test_labels.bin", 10000);
	std::cout << "Test data loaded." << std::endl;

	// ── DIAGNOSTICS ─────────────────────────────────────────────────────────
	{
		const auto& [img0, lbl0] = training_data[0];
		const auto& [img1, lbl1] = training_data[1];

		// 1. Pixel value range — should be [0, 1] if normalised
		float pmin = *std::min_element(img0.rix.begin(), img0.rix.end());
		float pmax = *std::max_element(img0.rix.begin(), img0.rix.end());
		std::cout << "Train img[0] pixel range : [" << pmin << ", " << pmax << "]\n";

		// 2. One-hot label — exactly one element should be 1.0, the rest 0.0
		std::cout << "Train label[0] (one-hot) : ";
		for (size_t i = 0; i < 10; i++) std::cout << lbl0[i] << " ";
		std::cout << "\n";

		std::cout << "Train label[1] (one-hot) : ";
		for (size_t i = 0; i < 10; i++) std::cout << lbl1[i] << " ";
		std::cout << "\n";

		// 3. Test label — should be a plain integer 0-9
		const auto& [timg0, tlbl0] = test_data[0];
		const auto& [timg1, tlbl1] = test_data[1];
		std::cout << "Test  label[0] (int)     : " << tlbl0 << "\n";
		std::cout << "Test  label[1] (int)     : " << tlbl1 << "\n";
	}
	std::cout << std::endl;
	// ── END DIAGNOSTICS ─────────────────────────────────────────────────────

	auto nn = Network({ 784, 30, 10 });
	std::cout << "Network initialized.\n" << std::endl;

	nn.SGD(training_data, 30, 10, 3.0f, test_data);

	return 0;
}

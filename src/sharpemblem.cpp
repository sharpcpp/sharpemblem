#include <sharpemblem.h>

int main(int argc, const char** argv)
{
    if (argc < 3) 
    {
		std::cerr << "Usage: " << argv[0] << " <input_image> <output_prefix>";
        return 1;
    }

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_UNCHANGED);

    if (image.empty())
    {
        std::cerr << "image is empty.";
        return 1;
    }

    try
    {
        cv::Mat source{};
        if (image.cols > 128 || image.rows > 128)
        {
			std::cout << "Resizing image to 128x128\n";
            cv::resize(image, source, cv::Size(std::min<int>(128, image.cols), std::min<int>(128, image.rows)), 0, 0, cv::INTER_NEAREST);
        }
        else 
        {
            source = std::move(image);
        }

        se::bf4::rect_processor processor(source);

        std::vector<se::bf4::rect> rectangles = processor.gen(40); // generate 40 rectangles, because battlefield 4 emblems only support up to 40 layers.
        
        save_rects_to_json(rectangles, std::format("{}_emblem.json", argv[2]), source.cols, source.rows);

        if (!cv::imwrite( std::format("{}_delta.png", argv[2]), processor.delta()))
        {
            std::cerr << "Failed to save preview.\n";
            return 1;
        }

        std::cout << "\nOK " << rectangles.size() << " rectangles.\n";
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Error: " << exception.what() << '\n';

        return 1;
    }

    return 0;
}
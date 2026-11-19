#include "rect_processor.h"

namespace se::bf4 {
    [[nodiscard]] auto rect_processor::rect_sum(const cv::Mat& mat, const cv::Rect& rect) const noexcept -> double
    {
        int x2 = rect.x + rect.width;
        int y2 = rect.y + rect.height;

        return mat.at<double>(y2, x2) - mat.at<double>(rect.y, x2) - mat.at<double>(y2, rect.x) + mat.at<double>(rect.y, rect.x);
    }
    auto rect_processor::build_intgrls() noexcept -> void
    {
        std::vector<cv::Mat> channels{};
        cv::split(_original, channels);

        cv::Mat valid_mask{};
        cv::compare(_mask, 0, valid_mask, cv::CMP_GT);

        for (int i = 0; i < 3; ++i) 
        {
            cv::Mat channel_64{};
            cv::Mat squared{};

            channels[i].convertTo(channel_64, CV_64F);

            channel_64.setTo(0, valid_mask == 0);

            cv::multiply(channel_64, channel_64, squared);

            cv::integral(channel_64, _sums[i], CV_64F);
            cv::integral(squared, _sums_sqr[i], CV_64F);
        }

        cv::Mat valid_mask_64{};
        valid_mask.convertTo(valid_mask_64, CV_64F, 1.0 / 255.0);

        cv::integral(valid_mask_64, _mask_integral, CV_64F);
    }

    [[nodiscard]] auto rect_processor::sum_colors(const cv::Rect& rect) const noexcept -> cv::Vec3d
    {
        cv::Vec3d sum{ 0, 0, 0 };
        for (int i = 0; i < 3; i++) 
        {
            sum[i] = rect_sum(_sums[i], rect);
        }

        return sum;
    }
    [[nodiscard]] auto rect_processor::avg_color(const cv::Rect& rect) const -> cv::Vec3b
    {
        const double area = rect_sum(_mask_integral, rect);

        if (area <= 0.0) 
        {
            throw std::runtime_error("empty rect, or no pixels.");
        }

        const cv::Vec3d sum = sum_colors(rect);

        cv::Vec3b color{};

        for (int i = 0; i < 3; ++i) 
        {
            color[i] = cv::saturate_cast<uint8_t>(std::round(sum[i] / area));
        }

        return color;
    }
    [[nodiscard]] auto rect_processor::rect_inside_mask(const cv::Rect& rect) const noexcept -> bool
    {
        if (rect.empty() || rect.x < 0 || rect.y < 0 || rect.x > _mask.cols - rect.width || rect.y > _mask.rows - rect.height)
        {
            return false;
        }

        const cv::Mat roi = _mask(rect);

        return cv::countNonZero(roi) == rect.area();
    }
    [[nodiscard]] auto rect_processor::best_color(const cv::Rect& rect) const noexcept -> double
    {
        cv::Vec3d sum = sum_colors(rect);
        double sum_sqr = 0.0;
        double area = rect_sum(_mask_integral, rect);

        if (area <= 0.0) 
        {
            return {};
        }

        for (int i = 0; i < 3; i++) 
        {
            sum_sqr += rect_sum(_sums_sqr[i], rect);
        }

        double error = sum_sqr;

        for (int i = 0; i < 3; i++) 
        {
            error -= sum[i] * sum[i] / area;
        }

        return std::max(0.0, error);
    }
    auto rect_processor::rebuild_error_intgrl() noexcept -> void
    {
        cv::Mat error_map(_original.rows, _original.cols, CV_64F);

        for (int y = 0; y < _original.rows; ++y) {
            const cv::Vec3b* original_row = _original.ptr<cv::Vec3b>(y);
            const cv::Vec3b* avg_row = _delta.ptr<cv::Vec3b>(y);
            double* error_row = error_map.ptr<double>(y);

            for (int x = 0; x < _original.cols; ++x) 
            {
                if (_mask.at<uint8_t>(y, x) == 0)
                {
                    error_row[x] = 0.0;
                    continue;
                }

                double error = 0.0;
                for (int i = 0; i < 3; i++) {
                    double difference = static_cast<double>(original_row[x][i]) - avg_row[x][i];
                    error += difference * difference;
                }

                error_row[x] = error;
            }
        }

        cv::integral(error_map, _error_intgrl, CV_64F);
    }

    [[nodiscard]] auto rect_processor::rect_sum4(const cv::Mat& mat, int x, int y, int width, int height) const noexcept -> __m256d
    {
        const double* top = mat.ptr<double>(y);
        const double* bottom = mat.ptr<double>(y + height);
        __m256d tl = _mm256_loadu_pd(top + x);
        __m256d tr = _mm256_loadu_pd(top + x + width);
        __m256d bl = _mm256_loadu_pd(bottom + x);
        __m256d br = _mm256_loadu_pd(bottom + x + width);

        return _mm256_add_pd(_mm256_sub_pd(_mm256_sub_pd(br, tr), bl), tl);
    }

    [[nodiscard]] auto rect_processor::candidate_gains4(int x, int y, int width, int height) const noexcept -> __m256d
    {
        __m256d current_error = rect_sum4(_error_intgrl, x, y, width, height);

        __m256d area = rect_sum4(_mask_integral, x, y, width, height);

        __m256d expected_area = _mm256_set1_pd(static_cast<double>(width) * height);

        __m256d safe_area = _mm256_max_pd(area, _mm256_set1_pd(1.0));

        __m256d squared_error = _mm256_setzero_pd();

        for (int i = 0; i < 3; ++i)
        {
            __m256d sum = rect_sum4(_sums[i], x, y, width, height);
            __m256d sum_squared = rect_sum4(_sums_sqr[i], x, y, width, height);
            __m256d mean_error = _mm256_div_pd(_mm256_mul_pd(sum, sum), safe_area);

            __m256d channel_error = _mm256_sub_pd(sum_squared, mean_error);

            squared_error = _mm256_add_pd(squared_error, channel_error);
        }

        __m256d new_error = _mm256_max_pd(squared_error, _mm256_setzero_pd());

        __m256d gain = _mm256_sub_pd(current_error, new_error);

        __m256d valid = _mm256_cmp_pd(area, expected_area, _CMP_EQ_OQ);

        return _mm256_blendv_pd(_mm256_set1_pd(-std::numeric_limits<double>::infinity()), gain, valid);
    }
    [[nodiscard]] auto rect_processor::find_best_candidate() const noexcept -> candidate_rect
    {
        candidate_rect best{};

        const int image_width = _original.cols;
        const int image_height = _original.rows;

        alignas(sizeof(double) * 4) double gains[4];


        for (int height = 0; height < image_height; height += 2)
        {
            if (height > image_height) {
                continue;
            }

			for (int width = 0; width < image_width; width += 2)
            {
                if (width > image_width) {
                    continue;
                }

                const int max_x = image_width - width;

                for (int y = 0; y + height <= image_height; ++y)
                {
                    int x = 0;

                    for (; x + 3 <= max_x; x += 4)
                    {
                        __m256d simd_gains = candidate_gains4(x, y, width, height);
                        _mm256_store_pd(gains, simd_gains);

                        for (int lane = 0; lane < 4; ++lane)
                        {
                            if (!std::isfinite(gains[lane]) || gains[lane] <= best.gain())
                            {
                                continue;
                            }

                            const cv::Rect rect{ x + lane, y, width, height };

                            if (!rect_inside_mask(rect))
                                continue;

                            const cv::Vec3b color = avg_color(rect);

                            best.rect().bounds(rect);
                            best.rect().color(color);
                            best.gain(gains[lane]);
                        }
                    }

                    for (; x <= max_x; ++x)
                    {
                        const cv::Rect rect(x, y, width, height);

                        if (!rect_inside_mask(rect))
                            continue;

                        const double current_error = rect_sum(_error_intgrl, rect);

                        const double new_error = best_color(rect);

                        const double gain = current_error - new_error;

                        if (!std::isfinite(gain) || gain <= best.gain())
                        {
                            continue;
                        }

                        best.rect().bounds(rect);
                        best.rect().color(avg_color(rect));
                        best.gain(gain);
                    }
                }
            }
        }

        return best;
    }
    rect_processor::rect_processor(const cv::Mat& source)
    {
        if (source.empty()) 
        {
            throw std::runtime_error("Input image is empty.");
        }

        cv::Mat posterize_input{};

        if (source.channels() == 4)
        {
            cv::Mat bgra{};
            source.convertTo(bgra, CV_8UC4);

            cv::Mat alpha{};
            cv::extractChannel(bgra, alpha, 3);

            cv::compare(alpha, 0, _mask, cv::CMP_GT);

            cv::cvtColor(bgra, posterize_input, cv::COLOR_BGRA2BGR);

        }
        else if (source.channels() == 3)
        {
            source.convertTo(posterize_input, CV_8UC3);

            _mask = cv::Mat(source.size(), CV_8UC1, cv::Scalar(255));
        }
        else if (source.channels() == 1)
        {
            cv::cvtColor(source, posterize_input, cv::COLOR_GRAY2BGR);

            _mask = cv::Mat(source.size(), CV_8UC1, cv::Scalar(255));
        }
        else
        {
            throw std::runtime_error("Unsupported number of image channels.");
        }

		_original = posterize_input;//posterize(posterize_input, 12); #TODO add posterization option, complex images require posterization to reduce the number of colors, but simple images do not need it.

        _delta = cv::Mat::zeros(_original.size(), CV_8UC3);

        build_intgrls();
        rebuild_error_intgrl();
    }

    [[nodiscard]] auto rect_processor::posterize(const cv::Mat& input, int colorCount) -> cv::Mat {
        cv::Mat alpha{};
        cv::Mat colors{};


        bool has_alpha;
        if ((has_alpha = input.channels() == 4)) 
        {
            cv::Mat channels[4];
            cv::split(input, channels);
            alpha = channels[3];
            cv::merge(std::vector<cv::Mat>{ channels[0], channels[1], channels[2] }, colors);

        }
        else 
        {
            colors = input.clone();
        }

        cv::Mat cluster{};
        cv::cvtColor(colors, cluster, cv::COLOR_BGR2Lab);

        std::vector<cv::Vec3b> pixels{};
        pixels.reserve(cluster.total());

        for (int y = 0; y < cluster.rows; y++) 
        {
            for (int x = 0; x < cluster.cols; x++) 
            {
                if (has_alpha && alpha.at<uint8_t>(y, x) == 0) 
                {
                    continue;
                }

                pixels.emplace_back(cluster.at<cv::Vec3b>(y, x));
            }
        }

        if (pixels.empty())
        {
            return input.clone();
        }

        int k = std::min(colorCount, static_cast<int>(pixels.size()));
        std::vector<cv::Vec3b>::size_type size = pixels.size();
        cv::Mat samples(size, 3, CV_32F);
        for (int i = 0; i < size; i++) {
            samples.at<float>(i, 0) = pixels[i][0];
            samples.at<float>(i, 1) = pixels[i][1];
            samples.at<float>(i, 2) = pixels[i][2];
        }

        cv::Mat common_color_mat{};
        cv::Mat center_color_mat{};

        cv::kmeans(samples,
            k,
            common_color_mat,
            cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.5),
            3,
            cv::KMEANS_PP_CENTERS, center_color_mat);

        cv::Mat quantized = cluster.clone();
        for (int y = 0, idx = 0; y < cluster.rows; y++) 
        {
            for (int x = 0; x < cluster.cols; x++) 
            {
                if (has_alpha && alpha.at<uint8_t>(y, x) == 0) 
                {
                    continue;
                }

                int cl = common_color_mat.at<int>(idx++, 0);
                quantized.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    cv::saturate_cast<uint8_t>(center_color_mat.at<float>(cl, 0)),
                    cv::saturate_cast<uint8_t>(center_color_mat.at<float>(cl, 1)),
                    cv::saturate_cast<uint8_t>(center_color_mat.at<float>(cl, 2)));
            }
        }

        cv::Mat quantized_bgr{};
        cv::cvtColor(quantized, quantized_bgr, cv::COLOR_Lab2BGR);

        if (has_alpha) {
            cv::Mat output_mats[3];
            cv::split(quantized_bgr, output_mats);

            std::vector<cv::Mat> output{ output_mats[0], output_mats[1], output_mats[2] };

            cv::Mat result{};
            cv::merge(output, result);
            return result;
        }

        return quantized_bgr;

    }

    [[nodiscard]] auto rect_processor::gen(int nrects) -> std::vector<se::bf4::rect>
    {
        std::vector<se::bf4::rect> rectangles{};

        while (static_cast<int>(rectangles.size()) < nrects)
        {
            candidate_rect best = find_best_candidate();
            if (best.gain() <= 1e-6) 
            {
                break;
            }

			se::bf4::rect& rectangle = rectangles.emplace_back(best.rect().bounds(), best.rect().color());

            paint(rectangle);

            rebuild_error_intgrl();

            std::cout << "[" << rectangle.x()
                << ", " << rectangle.y()
                << ", " << rectangle.w()
                << ", " << rectangle.h()
                << "], BGR=("
                << static_cast<int>(rectangle.color()[0]) << ", "
                << static_cast<int>(rectangle.color()[1]) << ", "
                << static_cast<int>(rectangle.color()[2]) << ")"
                << ", G=" << best.gain() << " \t\t\t[" << std::roundf(static_cast<float>(rectangles.size()) / nrects * 100) << "%]"
                << '\n';
        }

        cv::cvtColor(_delta, _output, cv::COLOR_BGR2BGRA);
        cv::insertChannel(_mask, _output, 3);

        return rectangles;
    }
}
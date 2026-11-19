#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <immintrin.h>

#include <bf4/rect.h>
#include <bf4/candidate_rect.h>

namespace se::bf4 {
    class rect_processor
    {
    private:
        cv::Mat _original;
        cv::Mat _delta;
        cv::Mat _mask;

        std::array<cv::Mat, 3> _sums;
        std::array<cv::Mat, 3> _sums_sqr;

        cv::Mat _error_intgrl;
        cv::Mat _mask_integral;
        cv::Mat _output;
    private:

        [[nodiscard]] auto rect_sum(const cv::Mat& mat, const cv::Rect& rect) const noexcept -> double;
        auto build_intgrls() noexcept -> void;
        [[nodiscard]] auto sum_colors(const cv::Rect& rect) const noexcept -> cv::Vec3d;

        [[nodiscard]] auto avg_color(const cv::Rect& rect) const -> cv::Vec3b;
        [[nodiscard]] auto rect_inside_mask(const cv::Rect& rect) const noexcept -> bool;
        [[nodiscard]] auto best_color(const cv::Rect& rect) const noexcept -> double;
        auto rebuild_error_intgrl() noexcept -> void;
        [[nodiscard]] auto rect_sum4(const cv::Mat& mat, int x, int y, int width, int height) const noexcept -> __m256d;
        [[nodiscard]] auto candidate_gains4(int x, int y, int width, int height) const noexcept -> __m256d;
        [[nodiscard]] auto find_best_candidate() const noexcept -> candidate_rect;
        auto paint(const se::bf4::rect& rectangle) -> void { cv::rectangle(_delta, rectangle.bounds(), rectangle.color_as_scalar(), cv::FILLED); }
        [[nodiscard]] auto posterize(const cv::Mat& input, int colorCount = 12) -> cv::Mat;
    public:
        explicit rect_processor(const cv::Mat& source);
        [[nodiscard]] auto gen(int nrects) -> std::vector<se::bf4::rect>;
        [[nodiscard]] auto delta() -> const cv::Mat& { return _output; }
    };
}
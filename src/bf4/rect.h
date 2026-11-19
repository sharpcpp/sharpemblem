#pragma once
#include <opencv2/opencv.hpp>

namespace se::bf4 {
    class rect final
    {
        using underlying_rectangle_type = cv::Rect;
        using size_type = typename underlying_rectangle_type::value_type;
        using underlying_color_type = cv::Vec3b;
    public:
        rect(underlying_rectangle_type rect, underlying_color_type color) : _bounds(std::move(rect)), _color(std::move(color)) {}
        explicit rect() : _bounds(), _color() {}
    public:
        [[nodiscard]] auto x() const noexcept -> size_type { return _bounds.x; }
        [[nodiscard]] auto y() const noexcept -> size_type { return _bounds.y; }
        [[nodiscard]] auto w() const noexcept -> size_type { return _bounds.width; }
        [[nodiscard]] auto h() const noexcept -> size_type { return _bounds.height; }
        [[nodiscard]] auto bounds() const noexcept -> const underlying_rectangle_type& { return _bounds; }
        auto bounds(underlying_rectangle_type bounds) noexcept -> void { _bounds = std::move(bounds); }
        [[nodiscard]] auto color_as_scalar() const noexcept -> cv::Scalar { return cv::Scalar(_color[0], _color[1], _color[2]); }
        [[nodiscard]] auto color() const noexcept -> const underlying_color_type& { return _color; }
        auto color(underlying_color_type color) noexcept -> void { _color = std::move(color); }
    private:
        underlying_rectangle_type _bounds;
        underlying_color_type _color;
    };
}
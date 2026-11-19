#pragma once

#include <iostream>
#include <bf4/rect_processor.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>

static auto to_hex_color(const cv::Vec3b& color) noexcept -> std::string
{
    char buffer[8];

    std::snprintf(
        buffer,
        sizeof(buffer),
        "#%02X%02X%02X",
        color[2],
        color[1],
        color[0]
    );

    return buffer;
}

[[nodiscard]] static auto rects_to_json(const std::vector<se::bf4::rect>& rectangles, int cols, int rows) -> nlohmann::json
{
    nlohmann::json result = nlohmann::json::object();
    result["objects"] = nlohmann::json::array();

    for (const se::bf4::rect& rectangle : rectangles)
    {
        double scale_x = 320.0 / static_cast<float>(cols);
        double scale_y = 320.0 / static_cast<float>(rows);

        nlohmann::json item = {
            { "angle", 0 },
            { "asset", "Square" },
            { "fill", to_hex_color(rectangle.color()) },
            { "flipX", false },
            { "flipY", false },
            { "height", rectangle.h() * scale_y },
            { "left", rectangle.x() * scale_x + rectangle.w() * scale_x / 2 },
            { "opacity", 1 },
            { "selectable", true },
            { "top", rectangle.y() * scale_y + rectangle.h() * scale_y / 2 },
            { "width", rectangle.w() * scale_x }
        };
        result["objects"].push_back(item);
    }

    return result;
}

static void save_rects_to_json(const std::vector<se::bf4::rect>& rectangles, const std::string& filename, int cols, int rows)
{
    nlohmann::json result = rects_to_json(rectangles, cols, rows);

    std::ofstream file(filename);

    file << std::setw(4) << result << '\n';
}
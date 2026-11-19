#include <bf4/rect.h>

namespace se::bf4 {
    class candidate_rect final
    {
    public:
        candidate_rect(rect&& r, double g) : _rect(std::move(r)), _gain(g) {}
        candidate_rect() : _rect(), _gain() {}
    public:
        [[nodiscard]] constexpr auto gain() const noexcept -> double { return _gain; }
        auto gain(double g) noexcept -> void { _gain = g; }
        [[nodiscard]] auto rect() noexcept -> rect& { return _rect; }
    private:
        se::bf4::rect _rect;
        double _gain;
    };
}
#include <iostream>
#include <random>

#include "perlin.hpp"

using std::random_device;

float smoothstep(float value) { return {}; }

using perlin_fbm = pln::fbm<pln::perlin<float>>;

int main(int, char**)
{
    pln::perlin<float> perlin(0xffdd3u);
    float value = perlin.noise(12.0f, 12.0f);

    // bfm = Brownian Fractional Motion, 
    // usamos o movimento browniano para acumular os ruidos
    // baseados em determinados parametros, ex: freq, amplitude e etcc.
    float result = pln::fbm<pln::perlin<float>>::accumulate(1.0f, 2.0f, 3.0f);

    std::cout << result << '\n';
    std::cout << perlin.noise(5.0f, 2.0f, true) << '\n';
    return {};
}

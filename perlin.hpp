#pragma once

#include <array>
#include <random>
#include <concepts>
#include <type_traits>
#include <algorithm>
#include <functional>

namespace pln
{
    namespace detail
    {
        template <std::floating_point FloatType>
        static constexpr inline FloatType map(FloatType value, FloatType old_min, 
                                              FloatType old_max, FloatType new_min, 
                                              FloatType new_max)
        {
            FloatType alpha{ (value - old_min) / (old_max - old_min) };
            return std::lerp(new_min, new_max, alpha);
        }
        
        // Caso especial do 'map', quando: (old_min, old_max) = (-1, 1) 
        // e (new_min, new_max) = (0, 1)
        template <std::floating_point FloatType>
        static constexpr inline FloatType remmap01(FloatType value)
        {
            return (value * FloatType{0.5}) + FloatType{0.5};
        }

        // No caso do 1D, o gradiente pode ser 1 e -1
        template <std::floating_point FloatType>
        static constexpr inline FloatType dot_grad(std::uint8_t hash, FloatType x)
        {
            return (hash & 0x1U) ? x : -x;
        }

        // No caso 2d, o gradiente pode ser qualquer um dos 8 vetores de direção
        // apontado para as arestas de um quadrado-unitário. 
        template <std::floating_point FloatType>
        static constexpr inline FloatType dot_grad(std::uint8_t hash, FloatType x, FloatType y)
        {
            switch (hash & 0x7U) 
            {
                case 0x0U: return  x + y;
                case 0x1U: return  x;
                case 0x2U: return  x - y;
                case 0x3U: return  -y;
                case 0x4U: return  -x - y;
                case 0x5U: return  -x;
                case 0x6U: return  -x + y;
                case 0x7U: return  y;
                default:  return {};
            }
        }

        // No caso 3d, o gradiente pode ser qualquer um dos 12 vetores de direção
        // apontado para as arestas de um cubo-unitário (arredondado para 16 com as duplicações). 
        template <std::floating_point FloatType>
        constexpr inline FloatType dot_grad(std::uint8_t hash, FloatType x, FloatType y, FloatType z)
        {
            switch(hash & 0xFU)
            {
                case 0x0U: return  x + y;
                case 0x1U: return -x + y;
                case 0x2U: return  x - y;
                case 0x3U: return -x - y;
                case 0x4U: return  x + z;
                case 0x5U: return -x + z;
                case 0x6U: return  x - z;
                case 0x7U: return -x - z;
                case 0x8U: return  y + z;
                case 0x9U: return -y + z;
                case 0xAU: return  y - z;
                case 0xBU: return -y - z;
                case 0xCU: return  y + x;
                case 0xDU: return -y + z;
                case 0xEU: return  y - x;
                case 0xFU: return -y - z;
                default: return {};
            }
        }

        template <typename GradientNoise, 
                  std::floating_point FloatType, 
                  typename... Args>
        static constexpr inline 
        FloatType fbm_accum(std::uint32_t noctaves, FloatType lacunarity, 
                            FloatType gain, FloatType amplitude, FloatType frequency,
                            const GradientNoise& grad_noise, bool normalize, bool remap,
                            Args&&... args)
        {
            FloatType result{};
            FloatType amplitude_accum{};

            for (; noctaves; --noctaves)
            {
                result += grad_noise.noise(frequency * std::decay_t<Args>(args)...) * amplitude;
                amplitude_accum += amplitude;
                amplitude *= gain;
                frequency *= lacunarity;
            }

            if (normalize)
                result /= amplitude_accum;

            if (remap)
                result = remmap01<FloatType>(result);

            return result;
        }

    }

    // Conceito sobre uma função de curva suave
    // uma função é dita suave quando para todos os pontos na curva
    // existe a derivada naquelo ponto, ou seja, a função tem que ser contínua
    // garantindo assim uma única lei de formação para aquela função
    template <typename FloatType>
    struct quintic_smoothstep_functor
    {
        constexpr FloatType operator()(FloatType t) const noexcept
        {
            return t * t * t * (t * (t * FloatType(6.0) - FloatType(15.0)) + FloatType(10.0));
        }
    };

    template <std::floating_point FloatType,
              typename PermTableType = std::array<std::uint8_t, 512>,
              typename DefaultRandomEngine = std::mt19937>
    class basic_gradient_noise
    {
    public:
        using value_type = FloatType;
        using perm_table_type = PermTableType;
        using default_rnd_engine = DefaultRandomEngine;
        using seed_type = typename default_rnd_engine::result_type;
        using smooth_func = std::function<FloatType(FloatType)>;

        constexpr basic_gradient_noise(const seed_type& seed) noexcept
            : m_seed{ seed }
        {
            reseed(m_seed);
        }

        constexpr seed_type get_seed() const noexcept
        {
            return m_seed;
        }

        constexpr const perm_table_type& get_perm_table() const noexcept
        {
            return m_perm_table;
        }

        void reseed(const seed_type& seed)
        {
            static default_rnd_engine rnd_engine{ seed };
            std::size_t half_length{ std::size(m_perm_table) >> 1u };

            auto start{ m_perm_table.begin() };
            auto end{ m_perm_table.begin() + half_length};

            std::iota(start, end, 0u);
            std::shuffle(start, end, rnd_engine);
            std::copy(start, end, end);
        }

        virtual FloatType noise(FloatType x, bool normalize, smooth_func&& fn) const noexcept = 0;
        virtual FloatType noise(FloatType x, FloatType y, bool normalize, smooth_func&& fn) const noexcept = 0;
        virtual FloatType noise(FloatType x, FloatType y, FloatType z, bool normalize, smooth_func&& fn) const noexcept = 0;

    protected:
        seed_type m_seed;
        perm_table_type m_perm_table;
    };

    template <std::floating_point FloatType = float>
    class perlin : public basic_gradient_noise<FloatType>
    {
    public:
        using Base = basic_gradient_noise<FloatType>;
        using Base::basic_gradient_noise;
        using typename Base::smooth_func;
        using Base::get_perm_table;

        // 1D perlin noise
        [[nodiscard]]
        FloatType noise(FloatType x, bool normalize = false, smooth_func&& smoothstep = quintic_smoothstep_functor<FloatType>{}) const noexcept override;

        // 2D perlin noise
        [[nodiscard]]
        FloatType noise(FloatType x, FloatType y, bool normalize = false, smooth_func&& smoothstep = quintic_smoothstep_functor<FloatType>{}) const noexcept override;

        // 3D perlin noise
        [[nodiscard]]
        FloatType noise(FloatType x, FloatType y, FloatType z, bool normalize = false, smooth_func&& smoothstep = quintic_smoothstep_functor<FloatType>{}) const noexcept override;
    };

    template <std::floating_point FloatType>
    FloatType perlin<FloatType>::noise(FloatType x, bool normalize, smooth_func&& smoothstep) const noexcept
    {
        using detail::dot_grad;
        using detail::remmap01;

        using ValueTypeDecayed = std::decay_t<FloatType>;
        using IntType = typename std::conditional_t<std::is_same_v<ValueTypeDecayed, float>, std::int32_t, std::int64_t>;
        auto&& fsmoothstep{ std::forward<smooth_func>(smoothstep) };

        // determinando as coordenadas da linha-unitária
        IntType x0 = std::floor(x);
        IntType x1 = x0 + 1;
        x -= x0; 

        // garantimos que esteja no intervalo [0, 255]
        x0 &= 255;
        x1 &= 255;

        // Determinamos o peso da interpolação aplicando uma função de curva suave
        // s-curve function
        FloatType sx{ fsmoothstep(x) };

        // determinamos tabela de permutação
        const auto& perm_table = get_perm_table();

        // geramos os valores hash para cada ponto na linha-unitária
        std::uint8_t a = perm_table[x0];
        std::uint8_t b = perm_table[x1];

        // Linearly interpolate between dot products of each gradient with its distance to the input location
        // Aplicamos uma interpolação linear no produto escalar de cada
        // vetor gradiente com a seu vetor de distancia em relação a 
        // posicão de entrada
        FloatType average = std::lerp(dot_grad(a, x),
                                      dot_grad(b, x - FloatType{1}),
                                      sx);

        // remapeamos o average no intervalo [-1, 1] para o intervalo [0, 1]
        if (normalize)
        {
            // antes de remapear, temos que garantir que o resultado do noise
            // esteja realmente entre [-1, 1], pois existem casos raros
            // em que o resultado pode não estar nesse intervalo
            average = std::clamp(average, FloatType{-1.0}, FloatType{1.0});
            average = remmap01<FloatType>(average);
        }

        return average;
    }

    template <std::floating_point FloatType>
    FloatType perlin<FloatType>::noise(FloatType x, FloatType y, bool normalize, smooth_func&& smoothstep) const noexcept
    {
        using detail::dot_grad;
        using detail::remmap01;

        using ValueTypeDecayed = std::decay_t<FloatType>;
        using IntType = typename std::conditional_t<std::is_same_v<ValueTypeDecayed, float>, std::int32_t, std::int64_t>;
        auto&& fsmoothstep{ std::forward<smooth_func>(smoothstep) };

        // determinando as coordenadas do quadrado-unitário
        // n01-------n11
        // | \     /  |
        // |  (x, y)  |
        // |  /    \  |
        // n00-------n10

        // definimos a parte inteira de x e y
        IntType xi = static_cast<IntType>(std::floor(x));
        IntType yi = static_cast<IntType>(std::floor(y));

        // definimos a parte fracionária de x e y
        FloatType xf = x - xi;
        FloatType yf = y - yi;

        // garantimos que esteja no intervalo [0, 255]
        xi &= 255;
        yi &= 255;

        // Determinamos o peso da interpolação aplicando uma função de curva suave
        // s-curve function
        FloatType sx{ fsmoothstep(xf) };
        FloatType sy{ fsmoothstep(yf) };

        // determinamos tabela de permutação
        const auto& perm_table = get_perm_table();

        // geramos os valores hash para cada ponto no quadrado-unitário
        // n_xy = vizinho do ponto na posição (x, y)
        std::uint8_t n00 = perm_table[perm_table[xi] + yi];
        std::uint8_t n10 = perm_table[perm_table[xi + 1] + yi];
        std::uint8_t n11 = perm_table[perm_table[xi + 1] + yi + 1];
        std::uint8_t n01 = perm_table[perm_table[xi] + yi + 1];

        // Aplicamos uma interpolação linear no produto escalar de cada
        // vetor gradiente com o seu vetor de distancia em relação a 
        // posicão de entrada
        FloatType a = std::lerp(dot_grad(n00, xf, yf), 
                                dot_grad(n10, xf - 1, yf - 1), 
                                sx);

        FloatType b = std::lerp(dot_grad(n01, xf, yf - 1), 
                                dot_grad(n11, xf - 1, yf - 1), 
                                sx);

        FloatType average = std::lerp(a, b, sy);

        // remapeamos o average no intervalo [-1, 1] para o intervalo [0, 1]
        if (normalize)
        {
            // antes de remapear, temos que garantir que o resultado do noise
            // esteja realmente entre [-1, 1], pois existem casos raros
            // em que o resultado pode não estar nesse intervalo
            average = std::clamp(average, FloatType{-1.0}, FloatType{1.0});
            average = remmap01<FloatType>(average);
        }

        return average;
    }

    template <std::floating_point FloatType>
    FloatType perlin<FloatType>::noise(FloatType x, FloatType y, FloatType z, bool normalize, smooth_func&& smoothstep) const noexcept
    {
        using detail::dot_grad;
        using detail::remmap01;

        using ValueTypeDecayed = std::decay_t<FloatType>;
        using IntType = typename std::conditional_t<std::is_same_v<ValueTypeDecayed, float>, std::int32_t, std::int64_t>;
        auto&& fsmoothstep{ std::forward<smooth_func>(smoothstep) };

        // determinando as coordenadas do cubo-unitário

        // definimos a parte inteira de (x, y, z)
        IntType xi = static_cast<IntType>(std::floor(x));
        IntType yi = static_cast<IntType>(std::floor(y));
        IntType zi = static_cast<IntType>(std::floor(z));

        // definimos a parte fracionária de (x, y, z)
        FloatType xf = x - xi;
        FloatType yf = y - yi;
        FloatType zf = z - zi;

        // garantimos que esteja no intervalo [0, 255]
        xi &= 255;
        yi &= 255;
        zi &= 255;

        // Determinamos o peso da interpolação aplicando uma função de curva suave
        // s-curve function
        FloatType sx{ fsmoothstep(xf) };
        FloatType sy{ fsmoothstep(yf) };
        FloatType sz{ fsmoothstep(zf) };

        // determinamos tabela de permutação
        const auto& perm_table = get_perm_table();

        // geramos os valores hash para cada ponto no quadrado-unitário
        // n_xyz = vizinho do ponto na posição (x, y, z)
        std::uint8_t n000 = perm_table[perm_table[perm_table[xi] + yi] + zi];
        std::uint8_t n001 = perm_table[perm_table[perm_table[xi] + yi] + zi + 1];
        std::uint8_t n010 = perm_table[perm_table[perm_table[xi] + yi + 1] + zi];
        std::uint8_t n011 = perm_table[perm_table[perm_table[xi] + yi + 1] + zi + 1];
        std::uint8_t n100 = perm_table[perm_table[perm_table[xi + 1] + yi] + zi];
        std::uint8_t n101 = perm_table[perm_table[perm_table[xi + 1] + yi] + zi + 1];
        std::uint8_t n110 = perm_table[perm_table[perm_table[xi + 1] + yi + 1] + zi];
        std::uint8_t n111 = perm_table[perm_table[perm_table[xi + 1] + yi + 1] + zi + 1];

        // Aplicamos uma interpolação linear no produto escalar de cada
        // vetor gradiente com o seu vetor de distancia em relação a 
        // posicão de entrada
        FloatType x11 = std::lerp(dot_grad(n000, xf, yf, zf), 
                                  dot_grad(n100, xf - 1, yf, zf), 
                                  sx);

        FloatType x12 = std::lerp(dot_grad(n010, xf, yf - 1, zf), 
                                  dot_grad(n110, xf - 1, yf - 1, zf), 
                                  sx);

        FloatType x21 = std::lerp(dot_grad(n001, xf, yf, zf - 1), 
                                  dot_grad(n101, xf - 1, yf, zf - 1), 
                                  sx);
        
        FloatType x22 = std::lerp(dot_grad(n011, xf, yf - 1, zf - 1), 
                                  dot_grad(n111, xf - 1, yf - 1, zf - 1), 
                                  sx);
        
        FloatType a  = std::lerp(x11, x12, sy);
        FloatType b  = std::lerp(x21, x22, sy);
        FloatType average = std::lerp(a, b, sz);
        
        // remapeamos o average no intervalo [-1, 1] para o intervalo [0, 1]
        if (normalize)
        {
            // antes de remapear, temos que garantir que o resultado do noise
            // esteja realmente entre [-1, 1], pois existem casos raros
            // em que o resultado pode não estar nesse intervalo
            average = std::clamp(average, FloatType{-1.0}, FloatType{1.0});
            average = remmap01<FloatType>(average);
        }

        return average;
    }

    template <typename GradientNoise>
    struct fbm
    {
        template <typename FloatType>
        static constexpr inline 
        FloatType accumulate(FloatType x, 
                             std::uint32_t noctaves = 8u, 
                             FloatType lacunarity = FloatType{2.0},
                             FloatType gain = FloatType{0.5},
                             FloatType amplitude = FloatType{1.0},
                             FloatType frequency = FloatType{1.0},
                             bool normalize = true,
                             bool remap = true,
                             const GradientNoise& grad_noise = GradientNoise{ std::random_device{}() })
        {
            return detail::fbm_accum(noctaves, lacunarity, gain, amplitude, frequency, grad_noise, normalize, remap, x);
        }

        template <typename FloatType>
        static constexpr inline 
        FloatType accumulate(FloatType x, FloatType y,
                             std::uint32_t noctaves = 8u, 
                             FloatType lacunarity = FloatType{2.0}, 
                             FloatType gain = FloatType{0.5},
                             FloatType amplitude = FloatType{1.0},
                             FloatType frequency = FloatType{1.0},
                             bool normalize = true,
                             bool remap = true,
                             const GradientNoise& grad_noise = GradientNoise{ std::random_device{}() })
        {
            return detail::fbm_accum(noctaves, lacunarity, gain, amplitude, frequency, grad_noise, normalize, remap, x, y);
        }

        template <std::floating_point FloatType>
        static constexpr inline 
        FloatType accumulate(FloatType x, FloatType y, FloatType z,
                             std::uint32_t noctaves = 8u, 
                             FloatType lacunarity = FloatType{2.0}, 
                             FloatType gain = FloatType{0.5},
                             FloatType amplitude = FloatType{1.0},
                             FloatType frequency = FloatType{1.0},
                             bool normalize = true,
                             bool remap = true,
                             const GradientNoise& grad_noise = GradientNoise{ std::random_device{}() })
        {
            return detail::fbm_accum(noctaves, lacunarity, gain, amplitude, frequency, grad_noise, normalize, remap, x, y, z);
        }
    };
}

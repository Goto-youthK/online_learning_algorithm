#ifndef LOGISTIC_HPP
#define LOGISTIC_HPP

#include "fobos.hpp"
#include <math.h>

class Logistic : public FOBOS {
public:
    Logistic(const float eta_, const float lambda_)
        : FOBOS(eta_, lambda_)
    {};

    // 損失項のロジスティック回帰によるwの更新
    void logistic(const fv_t &fv, const int y, float step) {
        float inner_product = dotproduct(fv); // wとxの内積(w・x)
        size_t fv_size = fv.size();

        for(size_t i = 0; i < fv_size; i++) {
            // fv(key, x)
            int key = fv[i].first;
            float x_i = fv[i].second;

            std::tr1::unordered_map<int, float>::iterator wit = w.find(key);

            float tmp = - y * exp(- y * inner_product) / (1.0 + exp(- y * inner_product)) * x_i;

            if(wit != w.end())
                wit->second -= step * tmp; // 更新
            else
                w[key] = - step * tmp; // 新規登録
        }
    };

    // logisticによる重みwの更新(FOBOS)
    void update(const int max_iter) {
        std::random_shuffle(examples.begin(), examples.end()); // 学習データを適当にかき混ぜる
        // 指定した更新回数だけ繰り返す
        for(int iter = 0; iter < max_iter; iter++) {
            std::vector< std::pair<fv_t, int> >::iterator it = examples.begin();

            // 学習データの次元数だけ繰り返す
            for(; it != examples.end(); it++) {
                fv_t fv = it->first; // fv(key, x)
                int y = it->second;

                logistic(fv, y, eta); // 損失項のロジスティック回帰によるwの更新
            }

            // 正則化項の閉じた形での最適解の計算
            l1_regularize(iter); // L1正則化によるwの更新
        }
    };



    float logistic_predict(const fv_t& fv) {
        float inner_product = dotproduct(fv); // wとxの内積(w・x)
        return 1.0 / (1.0 + exp(- inner_product));
    };

    // 予測の判定
    bool classify(const fv_t& fv, const int y) {
        if(y > 0)
            return logistic_predict(fv) > 0.5; // true(1) or false(-1)
        else
            return logistic_predict(fv) < 0.5; // true(1) or false(-1)
    };
};

#endif

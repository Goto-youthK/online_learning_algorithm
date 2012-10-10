#ifndef FOBOS_HPP
#define FOBOS_HPP

#include "util.hpp"
//#include <tr1/unordered_map>
#include "/opt/local/include/boost/tr1/tr1/unordered_map"
#include <ext/hash_map>
#include <math.h>

class FOBOS {
public:
    std::vector< std::pair<fv_t, int> > examples; // 学習データ
    std::tr1::unordered_map<int, float> w;        // 重みベクトル
    int exampleN;                                 // 学習データ数
    float eta;                                    // 更新ステップ
    float lambda;                                 // パラメータ

    FOBOS(const float eta_, const float lambda_) :
        exampleN(0),
        eta(eta_),
        lambda(lambda_)
    {};
    virtual ~FOBOS() {}



    // データの追加
    void add_example(const fv_t& fv, const int y) {
        exampleN++;
        examples.push_back(std::make_pair(fv, y)); // 末尾に追加
    };



    // wとxの内積(w・x)の計算
    float dotproduct(const fv_t& fv) {
        float m = 0.0; // 内積(w・x)
        size_t fv_size = fv.size();

        for (size_t i = 0; i < fv_size; i++) {
            // fv(key, x)
            int key = fv[i].first;
            float x_i = fv[i].second;

            std::tr1::unordered_map<int, float>::iterator wit = w.find(key);

            // 内積(w・x)の計算
            if (wit != w.end())
                m +=  x_i * wit->second;
        }

        return m;
    };

    // 損失項の劣勾配法によるwの更新
    void muladd(const fv_t &fv, const int y, float step) {
        for (size_t i = 0; i < fv.size(); i++) {
            // fv(key, x)
            int key = fv[i].first;
            float x_i = fv[i].second;

            std::tr1::unordered_map<int, float>::iterator wit = w.find(key);

            if (wit != w.end())
                wit->second += step * y * x_i; // 更新(w_(t) = w_(t-1) - eta * y * x)
            else
                w[key] = step * y * x_i; // 新規登録
        }
    };



    // 更新回数による更新ステップの更新
    float get_eta(const int iter) {
        return 1.0 / (1.0 + iter / examples.size());
    };

    // クリッピングによるwの更新
    float clip_by_zero(const float wit, const float lambda_hat) {
        // sign(wit) * |wit - lambda_hat|の計算
        if (wit > 0.0) {
            if (wit > lambda_hat)
                return wit - lambda_hat;
            else
                return 0.0; // スパース化
        }
        else {
            if (wit < -lambda_hat)
                return wit + lambda_hat;
            else
                return 0.0; // スパース化
        }
    };

    // L1正則化によるwの更新
    void l1_regularize(const int iter) {
        float lambda_hat = get_eta(iter) * lambda;

        std::tr1::unordered_map<int, float> tmp = w; // 重みベクトル
        std::tr1::unordered_map<int, float>::iterator it = tmp.begin();

        // 重みベクトルの次元数だけ繰り返す
        for (; it != tmp.end(); it++) {
            int key = it->first;
            std::tr1::unordered_map<int, float>::iterator wit = w.find(key);
            float aaa = wit->second;

            wit->second = clip_by_zero(wit->second, lambda_hat);// クリッピングによるwの更新

            if (fabsf(aaa) < lambda_hat) {
                w.erase(wit);
            }
        }
    };



    // 更新
    virtual void update(const int max_iter) = 0;
    // 識別
    virtual bool classify(const fv_t& fv, const int y) = 0;
};

#endif

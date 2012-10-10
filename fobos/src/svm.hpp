#ifndef SVM_HPP
#define SVM_HPP

#include "fobos.hpp"

class SVM : public FOBOS {
public:
    SVM(const float eta_, const float lambda_) : FOBOS(eta_, lambda_) {};

    // y(w・x)の計算
    float margin(const fv_t &fv, const int y) {
        return y * dotproduct(fv); // y(w・x)
    };

    // svmによる重みwの更新(FOBOS)
    void update(const int max_iter) {
        std::random_shuffle(examples.begin(), examples.end()); // 学習データを適当にかき混ぜる
        // 指定した更新回数だけ繰り返す
        for(int iter = 0; iter < max_iter; iter++) {
            std::vector<std::pair<fv_t, int> >::iterator it = examples.begin();

            // 学習データの次元数だけ繰り返す
            for(; it != examples.end(); it++) {
                fv_t fv = it->first; // fv(key, x)
                int y = it->second;

                // 予測が誤りの場合(y(w・x) <= 1.0)
                if(margin(fv, y) <= 1.0)
                    muladd(fv, y, eta); // 損失項の劣勾配法によるwの更新
            }

            // 正則化項の閉じた形での最適解の計算
            l1_regularize(iter); // L1正則化によるwの更新
        }
    };



    // 予測の判定
    bool classify(const fv_t& fv, const int y) {
        return margin(fv, y) > 0.0; // true(1) or false(-1)
    };
};

#endif

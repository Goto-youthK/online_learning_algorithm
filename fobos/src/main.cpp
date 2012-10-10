#include "svm.hpp"
#include "logistic.hpp"
#include "cmdline.h"

// 学習データの追加
void add_training_example(FOBOS& fobos, const std::string& training_filename)
{
    std::vector<fv_t> v;
    std::ifstream ifs(training_filename.c_str());

    // 学習データの読み込み
    while(ifs && !ifs.eof()) {
        std::string line;
        fv_t fv;
        int y;

        getline(ifs, line);
        int ret = parse_line(line, fv, y);// 各データをfv, yに分割

        if (ret == 0)
            fobos.add_example(fv, y);// データの追加
    }
};

//
void run(FOBOS& fobos,
		 const std::string& training_filename,
		 const std::string& test_filename,
		 const int max_iter)
{
    add_training_example(fobos, training_filename);// 学習データの追加
    std::cerr << "Finished reading training data..." << std::endl;

    fobos.update(max_iter); // svmあるいはlogisticによる重みwの更新
    std::cerr << "Finished updating parameter..." << std::endl;

    int tp = 0; // true positive
    int fn = 0; // false negative
    int fp = 0; // false positive
    int tn = 0; // true negative
    std::ifstream ifs(test_filename.c_str());

    // テストデータの読み込み
    while(!ifs.eof()) {
        std::string line;
        fv_t fv;
        int y;

        getline(ifs, line);
        int ret = parse_line(line, fv, y); // 各データをfv, yに分割
        if (ret != 0) continue;

        // 正しいデータ
        if (y > 0) {
            // 正クラスに分類
            if (fobos.classify(fv, y)) // svmあるいはlogisticによる識別
                tp++;
            // 負クラスに分類
            else
                fn++;
        }
        // 誤ったデータ
        else {
            // 正クラスに分類
            if (fobos.classify(fv, y)) // svmあるいはlogisticによる識別
                fp++;
            // 負クラスに分類
            else
                tn++;
        }
    }
    std::cerr << "Finished reading test data..." << std::endl;

    // 結果の表示
    std::cerr << "accuracy  : " << ((double) tp + tn) / (tp + fn + fp + tn) << std::endl; // 正確度
    std::cerr << "precision : " << ((double) tp) / (tp + fp) << std::endl; // 精度（適合率）
    std::cerr << "recall    : " << ((double) tp) / (tp + fn) << std::endl; // 再現率
};

int main(int argc, char **argv) {
    cmdline::parser a;
    a.add<std::string>("train", 0, "training filename", true); // 学習データ
    a.add<std::string>("test", 0, "test filename", true); // テストデータ
    a.add<std::string>("classifier", 'c', "classifier type", false, "svm", cmdline::oneof<std::string>("svm", "logistic")); // デフォルトはsvm

    a.add<double>("eta", 'e', "update step", false, 1.0); // 更新ステップ
    a.add<double>("lambda", 'l', "regularization parameter", false, 0.9); // パラメータ
    a.add<int>("max_iter", 'i', "number of max iteration", false, 10); // 最大更新回数
    a.add<bool>("vervose", 'v', "vervose option ", false, false);
    a.add("help", 0, "print this message");

    bool ok = a.parse(argc, argv);

    if (argc == 1 || a.exist("help")) {
        std::cerr << a.usage();
        return 0;
    }

    // オプションエラーチェック
    if (!ok) {
        std::cerr << a.error() << std::endl << a.usage();
        return 0;
    }

    // 識別器がsvmの場合
    if (a.get<std::string>("classifier") == "svm") {
        SVM svm(a.get<double>("eta"), a.get<double>("lambda"));
        run(svm, a.get<std::string>("train"), a.get<std::string>("test"), a.get<int>("max_iter"));
    }
    // 識別器がlogisticの場合
    else {
        Logistic logistic(a.get<double>("eta"), a.get<double>("lambda"));
        run(logistic, a.get<std::string>("train"), a.get<std::string>("test"), a.get<int>("max_iter"));
    }

    return 0;
};

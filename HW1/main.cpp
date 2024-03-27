# include <iostream>
# include <fstream>
# include <string>  
# include <sstream>
# include <eigen3/Eigen/Dense>
using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

Eigen::MatrixXd design_matrix(Eigen::MatrixXd features, int M) {
    double s = 0.1;
    int N = features.rows();
    int K = features.cols();
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(N, M * K);
    for(int n = 0; n < N; n++) {
        for(int k = 0; k < K * M; k++) {
            int f = k / M; // the index of the feature
            int j = k % M; // the index of the basis function
            double mu_j = 3 * (-M + 1 + 2 * (j - 1) * (M - 1) / (M - 2 +1e-9)) / M;
            phi(n, k) = j == 0 ? 1 : sigmoid((features(n, f) - mu_j) / s);
        }
    }
    // std::cout << phi.rows() << ", " << phi.cols() << std::endl;
    return phi;
}

int countRowsInCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    int count = 0;

    while (std::getline(file, line)) {
        ++count;
    }

    return count  - 1;
}

int main(){
    // read data ***********************//
    // read file from a csv file
    std::ifstream file;
    file.open("HW1.csv");
    std::string line;
    
    // ***************demo***************//
    ifstream demo_file;
    demo_file.open("HW1_demo.csv");
    //******************************//

    // Read and discard the header line
    std::getline(file, line);
    Eigen::MatrixXd targets(15817, 1);
    Eigen::MatrixXd features(15817, 11);


    int row = 0;
    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string data;
        int col = 0;
        while(std::getline(ss, data, ',')){
            if (col == 0) { // assuming the first column is the target
                targets(row, 0) = std::stod(data);
            } else { // the rest are features
                features(row, col-1) = std::stod(data);
            }
            col++;
        }
        row++;
    }
    file.close();

    //*************demo******************//
    // Read and discard the header line
    std::getline(demo_file, line);
    int demo_count = countRowsInCSV("HW1.csv");
    Eigen::MatrixXd demo_features(demo_count, 11);
    Eigen::MatrixXd demo_targets(demo_count, 1);

    row = 0;
    while(std::getline(demo_file, line)){
        std::stringstream ss(line);
        std::string data;
        int col = 0;
        while(std::getline(ss, data, ',')){
            if (col == 0) { // assuming the first column is the target
                demo_targets(row, 0) = std::stod(data);
            } else { // the rest are features
                demo_features(row, col-1) = std::stod(data);
            }
            col++;
        }
        row++;
    }
    demo_file.close();
    //*******************************//

    // Assuming features and targets are Eigen::MatrixXd
    Eigen::MatrixXd train_features = Eigen::MatrixXd::Zero(10000, 11);
    Eigen::MatrixXd train_targets = Eigen::MatrixXd::Zero(10000, 1);
    Eigen::MatrixXd test_features = Eigen::MatrixXd::Zero(5817, 11);
    Eigen::MatrixXd test_targets = Eigen::MatrixXd::Zero(5817, 1);
    
    for(int i = 0; i < 10000; i++) {
        for(int j = 0; j < 11; j++) {
            train_features(i, j) = features(i, j);
        }
        train_targets(i, 0) = targets(i, 0);
    }

    for(int i = 10000; i < 15817; i++) {
        for(int j = 0; j < 11; j++) {
            test_features(i - 10000, j) = features(i, j);
        }
        test_targets(i - 10000, 0) = targets(i, 0);
    }


    // Assuming train_features and test_features are Eigen::MatrixXd
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(11);
    Eigen::VectorXd std = Eigen::VectorXd::Zero(11);

    for(int i = 0; i < 11; i++) {
        double sum = train_features.col(i).sum();
        mean(i) = sum / 10000.;
        double sum_squared = (train_features.col(i).array() - mean(i)).square().sum();
        std(i) = sqrt(sum_squared / 10000); // Use Bessel's correction
    }

    for(int i = 0; i < 10000; i++) {
        for(int j = 0; j < 11; j++) {
            train_features(i, j) = (train_features(i, j) - mean(j)) / std(j);
        }
    }
    for(int i = 0; i < 5817; i++) {
        for(int j = 0; j < 11; j++) {
            test_features(i, j) = (test_features(i, j) - mean(j)) / std(j);
        }
    }


    //***************demo***************//
    for (int i = 0; i < demo_count; i++) {
        for (int j = 0; j < 11; j++) {
            demo_features(i, j) = (demo_features(i, j) - mean(j)) / std(j);
        }
    }
    //***********************************//

    double lambda_ = 0.1;
    double train_error; 
    double test_error;
    double train_error_regularization; 
    double test_error_regularization;
    double train_accuracy;
    double test_accuracy;
    double train_accuracy_regularization;
    double test_accuracy_regularization;
    ofstream outfile("result.csv");
    outfile << "M,Train Error,Train Accuracy,Test Error,Test Accuracy,Train Error Regularization,Train Accuracy Regularization,Test Error Regularization,Test Accuracy Regularization\n";
    ofstream outfile5("demo_result.csv");
    outfile5 << "M,Demo Accuracy, Demo Accuracy Regularization\n";
    ofstream outfile3("cross_validation.csv");
    outfile3 << "M,Validation Error\n";
    for (int M=1; M<=30; M++) {
        if (M == 5 || M == 10 || M == 15 || M == 20 || M == 25 || M == 30) {
            cout << "M: " << M << endl;
            Eigen::MatrixXd train_phi = design_matrix(train_features, M);
            Eigen::MatrixXd test_phi = design_matrix(test_features, M);
            Eigen::MatrixXd w = train_phi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(train_targets);
            Eigen::MatrixXd I = Eigen::MatrixXd::Identity(M * train_features.cols(), M * train_features.cols());
            Eigen::MatrixXd w2 = (train_phi.transpose() * train_phi + lambda_ * I).ldlt().solve(train_phi.transpose() * train_targets);
            Eigen::MatrixXd train_predictions = train_phi * w;
            Eigen::MatrixXd test_predictions = test_phi * w;
            Eigen::MatrixXd train_predictions_regularization = train_phi * w2;
            Eigen::MatrixXd test_predictions_regularization = test_phi * w2;
            train_error = (train_predictions - train_targets).array().square().sum() / 10000.;
            test_error = (test_predictions - test_targets).array().square().sum() / 5817.;
            train_error_regularization = (train_predictions_regularization - train_targets).array().square().sum() / 10000.;
            test_error_regularization = (test_predictions_regularization - test_targets).array().square().sum() / 5817.;
            
            //************demo**********//
            // plot the fitting curve of the third feature of the demo set
            Eigen::MatrixXd demo_phi = design_matrix(demo_features, M);
            Eigen::MatrixXd demo_predictions = demo_phi * w;
            Eigen::MatrixXd demo_predictions_regularization = demo_phi * w2;
            ofstream outfile4("demo_predictions_" + std::to_string(M) + ".csv");
            outfile4 << "danceability,Predictions,Predictions Regularization,Test_target\n";
            for (int i = 0; i < demo_count; i++) {
                outfile4 << demo_features(i, 2) << "," << demo_predictions(i, 0) << "," << demo_predictions_regularization(i, 0) << "," << demo_targets(i, 0) << "\n";
                outfile4.flush();
            }
            outfile4.close();
            //**************************//

            
            // create a target_temp to calculate the accuracy
            Eigen::MatrixXd train_targets_temp = train_targets;
            Eigen::MatrixXd test_targets_temp = test_targets;

            for(int i = 0; i < 10000; i++) {
                if(train_targets_temp(i, 0) == 0) {
                    train_targets_temp(i, 0) = 1;
                }
            }
            for(int i = 0; i < 5817; i++) {
                if(test_targets_temp(i, 0) == 0) {
                    test_targets_temp(i, 0) = 1;
                }
            }

            // ************demo************//
            // create a target_temp to calculate the accuracy
            Eigen::MatrixXd demo_targets_temp = demo_targets;
            for (int i = 0; i < demo_count; i++) {
                if (demo_targets_temp(i, 0) == 0) {
                    demo_targets_temp(i, 0) = 1;
                }
            }
            //*****************************//

            
            Eigen::MatrixXd train_diff = Eigen::MatrixXd::Zero(10000, 1);
            for (int i = 0; i < 10000; i++) {
                train_diff(i, 0) = abs((train_predictions(i, 0) - train_targets_temp(i, 0)) / train_targets_temp(i, 0));
            }
            train_accuracy = 1 - train_diff.sum() / 10000.;

            Eigen::MatrixXd test_diff = Eigen::MatrixXd::Zero(5817, 1);
            for (int i = 0; i < 5817; i++) {
                test_diff(i, 0) = abs((test_predictions(i, 0) - test_targets_temp(i, 0)) / test_targets_temp(i, 0));
            }
            test_accuracy = 1 - test_diff.sum() / 5817.;

            Eigen::MatrixXd train_diff_regularization = Eigen::MatrixXd::Zero(10000, 1);
            for (int i = 0; i < 10000; i++) {
                train_diff_regularization(i, 0) = abs((train_predictions_regularization(i, 0) - train_targets_temp(i, 0)) / train_targets_temp(i, 0));
            }
            train_accuracy_regularization = 1 - train_diff_regularization.sum() / 10000.;

            Eigen::MatrixXd test_diff_regularization = Eigen::MatrixXd::Zero(5817, 1);
            for (int i = 0; i < 5817; i++) {
                test_diff_regularization(i, 0) = abs((test_predictions_regularization(i, 0) - test_targets_temp(i, 0)) / test_targets_temp(i, 0));
            }
            test_accuracy_regularization = 1 - test_diff_regularization.sum() / 5817.;

            outfile << M << "," << train_error << "," << train_accuracy << "," << test_error << "," << test_accuracy << "," << train_error_regularization << "," << train_accuracy_regularization << "," << test_error_regularization << "," << test_accuracy_regularization << "\n";
            outfile.flush();
            cout << M << "," << train_error << "," << train_accuracy << "," << test_error << "," << test_accuracy << "," << train_error_regularization << "," << train_accuracy_regularization << "," << test_error_regularization << "," << test_accuracy_regularization << "\n";
            
            // ***********demo*****************//
            // calculate the accuracy of the demo set
            Eigen::MatrixXd demo_diff = Eigen::MatrixXd::Zero(demo_count, 1);
            Eigen::MatrixXd demo_diff_regularization = Eigen::MatrixXd::Zero(demo_count, 1);
            for (int i = 0; i < demo_count; i++) {
                demo_diff(i, 0) = abs((demo_predictions(i, 0) - demo_targets_temp(i, 0)) / demo_targets_temp(i, 0));
                demo_diff_regularization(i, 0) = abs((demo_predictions_regularization(i, 0) - demo_targets_temp(i, 0)) / demo_targets_temp(i, 0));
            }
            double demo_accuracy = 1 - demo_diff.sum() / demo_count;
            double demo_accuracy_regularization = 1 - demo_diff_regularization.sum() / demo_count;
            cout << "Demo Accuracy: " << demo_accuracy << ", Demo Accuracy Regularization: " << demo_accuracy_regularization << endl;
            outfile5 << M << "," << demo_accuracy << "," << demo_accuracy_regularization << "\n";
            outfile5.flush();
            //*********************************//
            
            
            //  plot the fitting curve of the third feature 
            string filename = "fitting_curve_" + std::to_string(M) + ".csv";
            ofstream outfile2(filename);
            outfile2 << "danceability,Predictions,Predictions Regularization,Test_target\n";
            // test set
            // for (int i = 0; i < 5817 ; i++) {
            //     outfile2 <<  test_features(i,2) << "," << test_predictions(i, 0) << "," << test_predictions_regularization(i, 0) << "," << test_targets(i,0)<< "\n";
            //     outfile2.flush();
            // }
            // train set
            for (int i = 0; i < 10000 ; i++) {
                outfile2 <<  train_features(i,2) << "," << train_predictions(i, 0) << "," << train_predictions_regularization(i, 0) << "," << train_targets(i,0)<< "\n";
                outfile2.flush();
            }
            outfile2.close();

            // solution 2
            // Eigen::MatrixXd curve_features = Eigen::MatrixXd::Zero(10000, 11);
            // double min = 0;
            // double max = 1;
            // for (int i = 0; i < 10000; i++) {
            //     curve_features(i,2) = i / 10000.;
            // }
            

            // Eigen::MatrixXd phi = design_matrix(curve_features, M);
            // Eigen::MatrixXd predictions = phi * w;
            // Eigen::MatrixXd predictions_regularization = phi * w2;
            // for (int i = 0; i < 10000; i++) {
            //     outfile2 << curve_features(i, 2) << "," << predictions(i, 0) << "," << predictions_regularization(i, 0) << "\n";
            // }
            // outfile2.close();

        }
        
        // *************************************************************************************//
        
        if (1) {
            ///////// apply five-fold cross-validation /////////////////////////
            Eigen::MatrixXd cross_train_features = Eigen::MatrixXd::Zero(8000, 11);
            Eigen::MatrixXd cross_train_targets = Eigen::MatrixXd::Zero(8000, 1);
            Eigen::MatrixXd cross_validation_features = Eigen::MatrixXd::Zero(2000, 11);
            Eigen::MatrixXd cross_validation_targets = Eigen::MatrixXd::Zero(2000, 1);
            double valid_error = 0;

            // first fold (0-8000, 8000-10000) ////////
            for(int i = 0; i < 8000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i, j) = train_features(i, j);
                }
                cross_train_targets(i, 0) = train_targets(i, 0);
            }
            for(int i = 8000; i < 10000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_validation_features(i - 8000, j) = train_features(i, j);
                }
                cross_validation_targets(i - 8000, 0) = train_targets(i, 0);
            }

            Eigen::MatrixXd cross_train_phi = design_matrix(cross_train_features, M);
            Eigen::MatrixXd cross_valid_phi = design_matrix(cross_validation_features, M);
            Eigen::MatrixXd cross_w = cross_train_phi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cross_train_targets);
            Eigen::MatrixXd cross_valid_predictions = cross_valid_phi * cross_w;
            valid_error += (cross_valid_predictions - cross_validation_targets).array().square().sum() / 2000.;

            // second fold (0-6000, 6000-8000, 8000-10000)
            for(int i = 0; i < 6000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i, j) = train_features(i, j);
                }
                cross_train_targets(i, 0) = train_targets(i, 0);
            }
            for(int i = 6000; i < 8000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_validation_features(i - 6000, j) = train_features(i, j);
                }
                cross_validation_targets(i - 6000, 0) = train_targets(i, 0);
            }
            for(int i = 8000; i < 10000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i-2000, j) = train_features(i, j);
                }
                cross_train_targets(i-2000, 0) = targets(i, 0);
            }

            cross_train_phi = design_matrix(cross_train_features, M);
            cross_valid_phi = design_matrix(cross_validation_features, M);
            cross_w = cross_train_phi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cross_train_targets);
            cross_valid_predictions = cross_valid_phi * cross_w;
            valid_error += (cross_valid_predictions - cross_validation_targets).array().square().sum() / 2000.;

            // third fold (0-4000, 4000-6000, 6000-10000)
             for(int i = 0; i < 4000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i, j) = train_features(i, j);
                }
                cross_train_targets(i, 0) = train_targets(i, 0);
            }
            for(int i = 4000; i < 6000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_validation_features(i-4000, j) = train_features(i, j);
                }
                cross_validation_targets(i - 4000, 0) = train_targets(i, 0);
            }
            for(int i = 6000; i < 10000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i-2000, j) = train_features(i, j);
                }
                cross_train_targets(i-2000, 0) = targets(i, 0);
            }

            cross_train_phi = design_matrix(cross_train_features, M);
            cross_valid_phi = design_matrix(cross_validation_features, M);
            cross_w = cross_train_phi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cross_train_targets);
            cross_valid_predictions = cross_valid_phi * cross_w;
            valid_error += (cross_valid_predictions - cross_validation_targets).array().square().sum() / 2000.;
            

            // fourth fold (0-2000,2000-4000, 4000-10000)
            for(int i = 0; i < 2000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i, j) = train_features(i, j);
                }
                cross_train_targets(i, 0) = train_targets(i, 0);
            }
            for(int i = 2000; i < 4000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_validation_features(i - 2000, j) = train_features(i, j);
                }
                cross_validation_targets(i - 2000, 0) = train_targets(i, 0);
            }
            for(int i = 6000; i < 10000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i-2000, j) = train_features(i, j);
                }
                cross_train_targets(i-2000, 0) = train_targets(i, 0);
            }
            
            cross_train_phi = design_matrix(cross_train_features, M);
            cross_valid_phi = design_matrix(cross_validation_features, M);
            cross_w = cross_train_phi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cross_train_targets);
            cross_valid_predictions = cross_valid_phi * cross_w;
            valid_error += (cross_valid_predictions - cross_validation_targets).array().square().sum() / 2000.;
        

            // fifth fold (0-2000,2000-10000)
            for(int i = 0; i < 2000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_validation_features(i, j) = train_features(i, j);
                }
                cross_train_targets(i, 0) = targets(i, 0);
            }
            for(int i = 2000; i < 10000; i++) {
                for(int j = 0; j < 11; j++) {
                    cross_train_features(i-2000, j) = train_features(i, j);
                }
                cross_train_targets(i-2000, 0) = train_targets(i, 0);
            }

            cross_train_phi = design_matrix(cross_train_features, M);
            cross_valid_phi = design_matrix(cross_validation_features, M);
            cross_w = cross_train_phi.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cross_train_targets);
            cross_valid_predictions = cross_valid_phi * cross_w;
            valid_error += (cross_valid_predictions - cross_validation_targets).array().square().sum() / 2000.;

            valid_error /= 5;
            cout<< M << " " << valid_error << endl;
            outfile3 << M << "," << valid_error << "\n";
        } // end if
    } // end for
    outfile.close();
    outfile3.close();
    outfile5.close();
    return 0;
}

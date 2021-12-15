//
//  main.cpp
//  DensityEstimationTrees
//
//  Created by Xiaoyi Gu on 5/4/21.
//
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

//#include "dTree_utils.h"
#include "cf_det_utils.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::det;

BINDING_NAME("Conformal DET");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of density estimation trees for the density estimation "
    "task.  Density estimation trees can be trained or used to predict the "
    "density at locations given by query points.");

// Long description.
BINDING_LONG_DESC(
    "This program performs a number of functions related to Density Estimation "
    "Trees.  The optimal Density Estimation Tree (DET) can be trained on a set "
    "of data (specified by " + PRINT_PARAM_STRING("training") + ") using "
    "cross-validation (with number of folds specified with the " +
    PRINT_PARAM_STRING("folds") + " parameter).  This trained density "
    "estimation tree may then be saved with the output parameter."
    "\n\n"
    "The variable importances (that is, the feature importance values for each "
    "dimension) may be saved with the output"
    " parameter, and the density estimates for each training point may be saved"
    " with the " + PRINT_PARAM_STRING("training_set_estimates") + " output "
    "parameter."
    "\n\n"
    "Enabling path printing for each node outputs the path from the root node "
    "to a leaf for each entry in the test set, or training set (if a test set "
    "is not provided).  Strings like 'LRLRLR' (indicating that traversal went "
    "to the left child, then the right child, then the left child, and so "
    "forth) will be output. If 'lr-id' or 'id-lr' are given as the parameter, then the ID (tag) of "
    "every node along the path will be printed after or before the L or R "
    "character indicating the direction of traversal, respectively."
    "\n\n"
    "This program also can provide density estimates for a set of test points, "
    "specified in the " + PRINT_PARAM_STRING("test") + " parameter.  The "
    "density estimation tree used for this task will be the tree that was "
    "trained on the given training points, or a tree given as the parameter.  The density estimates for the test"
    " points may be saved using the " +
    PRINT_PARAM_STRING("test_set_estimates") + " output parameter.");

// See also...
BINDING_SEE_ALSO("Density estimation tree (DET) tutorial",
        "@doxygen/dettutorial.html");
BINDING_SEE_ALSO("Density estimation on Wikipedia",
        "https://en.wikipedia.org/wiki/Density_estimation");
BINDING_SEE_ALSO("Density estimation trees (pdf)",
        "http://www.mlpack.org/papers/det.pdf");
BINDING_SEE_ALSO("mlpack::tree::DTree class documentation",
        "@doxygen/classmlpack_1_1det_1_1DTree.html");

// Input data files.
PARAM_MATRIX_IN("train", "conformal training.", "t");
PARAM_MATRIX_IN("validation", "conformal validation",
    "i");

// Input or output model.
//PARAM_MODEL_IN(conf_dTree, "input_model", "Trained density estimation "
//    "tree to load.", "m");
//PARAM_MODEL_OUT(conf_dTree, "output_model", "Output to save trained "
//    "density estimation tree to.", "M");

// Output data files.
PARAM_MATRIX_IN("test", "conformal test.",
    "T");
PARAM_MATRIX_OUT("val_estimates", "validation estimates.", "e");
PARAM_MATRIX_OUT("test_estimates", "test estimates.", "E");

PARAM_MATRIX_OUT("train_time", "Training time of conformal", "Q");
PARAM_MATRIX_OUT("query_time", "Querying time of conformal", "q");

// Tagging and path printing options

PARAM_STRING_IN("criterion", "The loss function", "o", "NLL");
PARAM_STRING_IN("method", "conformal method", "a", "mll");
PARAM_STRING_IN("type", "emprical process type", "b", "sym");

PARAM_COL_IN("alpha_list", "list of alpha values", "p");

// Parameters for the training algorithm.
PARAM_INT_IN("folds", "The number of folds of cross-validation", "f", 10);
PARAM_INT_IN("min_leafs", "The minimum size of a leaf", "l", 5);
PARAM_INT_IN("max_leafs", "The maximum size of a leaf", "L", 10);

PARAM_DOUBLE_IN("delta", "The user specified delta level", "d", 0.01);
PARAM_INT_IN("mtry", "The user specified delta level", "m", 1);

PARAM_MODEL_IN(DTree<>, "train_model", "Trained density estimation "
    "tree to load.", "D");

static void mlpackMain() {
    // Validate input parameters.
//    RequireOnlyOnePassed({ "training", "input_model" }, true);

//    ReportIgnoredParam({{ "training", false }}, "training_set_estimates");
//    ReportIgnoredParam({{ "training", false }}, "folds");
//    ReportIgnoredParam({{ "training", false }}, "min_leaf_size");
//    ReportIgnoredParam({{ "training", false }}, "max_leaf_size");
//    ReportIgnoredParam({{ "training", false }}, "delta");
//    ReportIgnoredParam({{ "training", false }}, "loss");

    ReportIgnoredParam({{ "test", false }}, "test_set_estimates");

    RequireParamValue<int>("folds", [](int x) { return x >= 0; }, true,
        "folds must be non-negative");
    RequireParamValue<int>("max_leafs", [](int x) { return x > 0; }, true,
        "maximum leaf size must be positive");
    RequireParamValue<int>("min_leafs", [](int x) { return x > 0; }, true,
        "minimum leaf size must be positive");
    RequireParamValue<double>("delta", [](double x) { return x > 0; }, true,
        "tuning parameter must be positive");

    // Are we training a DET or loading from file?
    arma::mat trainingData;
    arma::mat validationData;
    arma::mat testData;
    DTree<arma::mat, int>* train_tree;
    trainingData = std::move(IO::GetParam<arma::mat>("train"));
    
    if (IO::HasParam("train_model")){
        train_tree = IO::GetParam<DTree<arma::mat>*>("train_model");
    }

    if (IO::HasParam("validation")){
        validationData = std::move(IO::GetParam<arma::mat>("validation"));
    }
    
    //    const bool regularization = CLI::HasParam("volume_regularization");
    const size_t maxLeafSize = IO::GetParam<int>("max_leafs");
    const size_t minLeafSize = IO::GetParam<int>("min_leafs");
    const double delta = IO::GetParam<double>("delta");
    
    size_t mtry;
    if(!IO::HasParam("mtry")){
        mtry = validationData.n_rows;
    }
    else{
        mtry = IO::GetParam<int>("mtry");
    }

    const string criterion = IO::GetParam<string>("criterion");
    const string method = IO::GetParam<string>("method");
    const string type = IO::GetParam<string>("type");
    size_t folds = IO::GetParam<int>("folds");
    // if (folds == 0)
    //     folds = trainingData.n_rows;

    // Obtain the optimal tree.
    

    Log::Info << "Method: " << method << endl;
    DTree<arma::mat, int>* tree;
    
    Timer::Start("det_training");

    if (IO::HasParam("alpha_list")){
        arma::vec alpha = IO::GetParam<arma::vec>("alpha_list");
        // if (IO::HasParam("train_model")){
        //     tree = FitConformal(train_tree, trainingData, validationData, method, delta, criterion, maxLeafSize, minLeafSize, folds, mtry, type);
        // }
        // else{
            tree = FitConformal(trainingData, validationData, method, delta, criterion, maxLeafSize, minLeafSize, folds, mtry, type);
        // }
    
    }
    else{
        // if (IO::HasParam("train_model")){
        //     tree = FitConformal(train_tree, trainingData, validationData, method, delta, criterion, maxLeafSize, minLeafSize, folds, mtry, type);
        // }
        // else{
            tree = FitConformal(trainingData, validationData, method, delta, criterion, maxLeafSize, minLeafSize, folds, mtry, type);
        // }
    }
    
    
    Timer::Stop("det_training");
    
    if (IO::HasParam("val_estimates")){
    // Compute density estimates for each point in the training set.
        arma::rowvec trainingDensities(validationData.n_cols);
        Timer::Start("det_estimation_time");
        for (size_t i = 0; i < validationData.n_cols; ++i)
          trainingDensities[i] = tree->ComputeValue(validationData.unsafe_col(i));
        Timer::Stop("det_estimation_time");

        IO::GetParam<arma::mat>("val_estimates") =
            std::move(trainingDensities);
    }

    //    else
    //    {
    //      tree = CLI::GetParam<conf_dTree*>("input_model");
    //    }

    // Compute the density at the provided test points and output the density in
    // the given file.
    if (IO::HasParam("test"))
    {
      testData = std::move(IO::GetParam<arma::mat>("test"));
      if (IO::HasParam("test_estimates"))
      {
        // Compute test set densities.
        Timer::Start("det_test_set_estimation");
        arma::rowvec testDensities(testData.n_cols);

        for (size_t i = 0; i < testData.n_cols; ++i)
        testDensities[i] = tree->ComputeValue(testData.unsafe_col(i));

        Timer::Stop("det_test_set_estimation");

        IO::GetParam<arma::mat>("test_estimates") = std::move(testDensities);

      }

      IO::GetParam<arma::mat>("train_time") = std::move(Timer::Get("det_training").count());
      IO::GetParam<arma::mat>("query_time") = std::move(Timer::Get("det_test_set_estimation").count());
    }

      // Compute training set estimates, if desired.

    // Save the model, if desired.
//    CLI::GetParam<conf_dTree*>("output_model") = tree;
}

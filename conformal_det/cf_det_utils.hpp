//
//  conf_dTree_utils.h
//  DensityEstimationTrees
//
//  Created by Xiaoyi Gu on 5/12/21.
//

#ifndef conf_dTree_utils_hpp
#define conf_dTree_utils_hpp

#include <mlpack/methods/det/dt_utils.hpp>
#include <iostream>
#include <armadillo>
#include <vector>
#include <string>
#include <cfloat>
#include <queue>
#include <assert.h>
#include <fstream>
#include <istream>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <boost/math/distributions/binomial.hpp>


using namespace std;
using namespace arma;
using namespace boost::math;
using namespace mlpack::det;

namespace mlpack {
namespace det {

struct CompareTrees{
    bool operator()(DTree<arma::mat, int>* t1, DTree<arma::mat, int>* t2){
        return (t1->GetReduction() < t2->GetReduction());
    }
};

double msQuantile(double alpha, std::string mode = "Con"){
    std::vector<double> sims;
    std::ifstream fin;
    if (mode == "Con"){
        fin.open("/Users/xgu1/Documents/DensityEstimationTrees/DensityEstimationTrees/simn1e4c.txt", std::ios_base::in);
    }
    else{
        std::ifstream fin("/Users/xgu1/Documents/DensityEstimationTrees/DensityEstimationTrees/simn1e4g.txt", std::ios_base::in);
    }
    
    double element;
    while (fin >> element){
        sims.push_back(element);
    }
    fin.close();
    
    size_t n = sims.size();
    std::sort(sims.begin(), sims.end());
    double h = (n+1)*(1-alpha);
    return sims[std::floor(h)] + (h - std::floor(h))*(sims[std::ceil(h)] - sims[std::floor(h)]);
};

void genIntv(size_t n, vector<size_t>& left, vector<size_t>& right){
    vector<int> l;
    vector<double> m_l;
    vector<int> d_l;
    
    double num;
    for (int i = 2; i <= std::floor(std::log((double) n/std::log(n))/std::log(2.0)); ++i){
        l.push_back(i);
        num = n*std::pow((double) 2,(-1)*i);
        
        m_l.push_back(num);
        d_l.push_back(std::ceil(num/(6.0*std::sqrt((double) i))));
    }
    
    int nintv = 0;
    int lb;
    int ub;
    for (size_t i = 0; i < d_l.size(); ++i){
        lb = std::ceil(m_l[i]/d_l[i]);
        ub = std::floor(2.0*m_l[i]/d_l[i]);
        for (size_t j = lb; j <= ub; ++j){
            nintv += std::floor((double) (n-1)/d_l[i]) - j + 1;
        }
    }
    
    left = std::vector<size_t>(nintv,0);
    right = std::vector<size_t>(nintv,0);
    size_t cnt = 0;
    double gridsize;
    for (size_t i = 0; i < d_l.size(); ++i){
        gridsize = d_l[i];
        lb = std::ceil(m_l[i]/gridsize)*gridsize;
        ub = std::floor(2.0*m_l[i]/gridsize)*gridsize;
        for (size_t j = 0; j <= std::floor((double) (n-1)/gridsize); ++j){
            for (size_t k = lb; k <= ub; k+=gridsize){
                if ((j*gridsize + k) <= n-1){
                    cnt += 1;
                    left[cnt-1] = j*gridsize;
                    right[cnt-1] = j*gridsize + k;
                }
            }
        }
    }
};

double FI(double l,
          double r,
          arma::Col<double> inverse_val_dens){
    arma::uvec inds = arma::find((l <= inverse_val_dens) && (inverse_val_dens <= r));
    return (double) inds.n_elem/inverse_val_dens.n_elem;
};

double ProbMass(arma::Col<double>& ctree_mass,
          std::unordered_set<int> s){
    double H = 0;
    for (int i : s){
        H += ctree_mass(i);
    }
    return H;
};

double logLR(size_t n,
             double H,
             double F){
    if (H == F){
        return 0.0;
    }

    if ((H == 0.0) || (H == 1.0)){
        return DBL_MAX;
    }

    return n*F*(std::log(F) - std::log(H)) + n*(1-F)*(std::log(1-F) - std::log(1-H));
};

double LL(double F){
    return std::sqrt(2.0 - 2.0*std::log(F*(1-F)));
};

double getArea(arma::Col<double> l1,
               arma::Col<double> u1,
               arma::Col<double> l2,
               arma::Col<double> u2){
    double area = 1.0;
    for (size_t d = 0; d < l1.n_elem; ++d){
        if ((l1(d) > u2(d)) || (l2(d) > u1(d))){
            return 0.0;
        }
        else{
            area *= (std::min(u1(d), u2(d)) - std::max(l1(d), l2(d)));
        }
    }
    return area;
};

double getMass(DTree<arma::mat, int>* tree,
               arma::Col<double> upper,
               arma::Col<double> lower){
    if (tree->SubtreeLeaves() == 1){
        double intersection = getArea(lower, upper, tree->MinVals(), tree->MaxVals());
        return intersection*(tree->Density());
    }
    else{
        return getMass(tree->Left(), upper, lower) + getMass(tree->Right(), upper, lower);
    }
};

double GenerateXi(const size_t N,
                  const size_t B,
                  const double del,
                  const double alp,
                  const std::string type){
    if (type == "sym"){
        int icdf = std::max(1, (int) quantile(boost::math::binomial(B, 1-alp), 1-del));
        arma::vec stats_vec(B);
        arma::vec unif_rvs;
        double max_unif;
        
        arma::arma_rng::set_seed_random();
        for (size_t i = 0; i < B; ++i){
            unif_rvs = arma::randu<vec>(N);
            unif_rvs = arma::sort(unif_rvs);
            max_unif = -DBL_MAX;
            for (size_t j = 0; j < N; ++j){
                max_unif = std::max(max_unif, std::sqrt((double) N)*((double) (j+1)/N - unif_rvs(j))/std::sqrt(unif_rvs(j)*(1 - unif_rvs(j))));
            }
            stats_vec(i) = max_unif;
        }
        stats_vec = arma::sort(stats_vec);
        return stats_vec(icdf-1);
    }
    else{
        int icdf = std::max(1, (int) quantile(boost::math::binomial(B, 1-alp), 1-del));
        arma::vec stats_vec(B);
        arma::vec unif_rvs;
        double max_unif1;
        double max_unif2;
        
        arma::arma_rng::set_seed_random();
        for (size_t i = 0; i < B; ++i){
            unif_rvs = arma::randu<vec>(N);
            unif_rvs = arma::sort(unif_rvs);
            max_unif1 = -DBL_MAX;
            max_unif2 = -DBL_MAX;
            for (size_t j = 0; j < N; ++j){
                max_unif1 = std::max(max_unif1, std::sqrt((double) N)*((double) (j+1)/N - unif_rvs(j))/std::sqrt(unif_rvs(j)*(1 - unif_rvs(j))));
                max_unif2 = std::max(max_unif2, std::sqrt((double) N)*(unif_rvs(j) - (double) j/N)/std::sqrt(unif_rvs(j)*(1 - unif_rvs(j))));
            }
            stats_vec(i) = std::max(max_unif1, max_unif2);
        }
        stats_vec = arma::sort(stats_vec);
        return stats_vec(icdf-1);
    }
};

void GenerateMLLConditions(const arma::Col<double>& inverse_train_dens,
                           const arma::Col<double>& inverse_val_dens,
                           const arma::Col<double>& unique_inverse_val_dens,
                           const std::vector<size_t>& intv_ind_left,
                           const std::vector<size_t>& intv_ind_right,
                           std::vector<double>& Fvals,
                           std::unordered_map<int, std::unordered_set<int>>& umap){
    double l;
    double r;
    double F;
    arma::uvec inds;
    std::unordered_set<int> uset;
    for (size_t i = 0; i < intv_ind_left.size(); ++i){
        l = unique_inverse_val_dens(intv_ind_left[i]);
        r = unique_inverse_val_dens(intv_ind_right[i]);
        F = FI(l, r, inverse_val_dens);
        Fvals.push_back(F);
        inds = arma::find((l <= inverse_train_dens) && (inverse_train_dens <= r));
        uset = {};
        uset.insert(inds.begin(), inds.end());
        umap.insert({i, uset});
    }
};

void UpdateTreeMass(DTree<arma::mat, int>*& node,
                    arma::vec& tree_mass,
                    std::vector<arma::Col<double>> upper,
                    std::vector<arma::Col<double>> lower,
                    std::unordered_map<int, std::unordered_set<int>>& umap){
    size_t n_lvl = upper.size();
    std::unordered_set<size_t> node_mask = node->GetMask();
    std::unordered_set<size_t> node_cond_mask = node->GetCondMask();
    std::unordered_set<size_t> node_left_mask;
    std::unordered_set<size_t> node_right_mask;
    std::unordered_set<size_t> node_left_condmask;
    std::unordered_set<size_t> node_right_condmask;
    
    double dens, area, dens_left, dens_right, area_left, area_right;
    for (int i : node_mask){
        dens = node->Density();
        area = getArea(lower[i], upper[i], node->MinVals(), node->MaxVals());
        tree_mass(i) -= dens*area;
        if (lower[i](node->SplitDim()) < node->SplitValue()){
            node_left_mask.insert(i);
            for (int j : node_cond_mask){
                if(umap[j].count(i) > 0){
                    node_left_condmask.insert(j);
                }
            }
            dens_left = (node->Left())->Density();
            area_left = getArea(lower[i], upper[i], (node->Left())->MinVals(), (node->Left())->MaxVals());
            tree_mass(i) += dens_left*area_left;
        }
        
        if (upper[i](node->SplitDim()) > node->SplitValue()){
            node_right_mask.insert(i);
            for (int j : node_cond_mask){
                if(umap[j].count(i) > 0){
                    node_right_condmask.insert(j);
                }
            }
            dens_right = (node->Right())->Density();
            area_right = getArea(lower[i], upper[i], (node->Right())->MinVals(), (node->Right())->MaxVals());
            tree_mass(i) += dens_right*area_right;
        }
    }
    
    (node->Left())->SetMask(node_left_mask);
    (node->Right())->SetMask(node_right_mask);
    (node->Left())->SetCondMask(node_left_condmask);
    (node->Right())->SetCondMask(node_right_condmask);
    
};

DTree<arma::mat, int>* FitMLL(arma::mat &train,
                              arma::mat &val,
                              const double delta = 0.01,
                              const std::string criterion = "NLL",
                              const size_t max_leaf_size = 10,
                              const size_t min_leaf_size = 5,
                              const size_t folds = 10,
                              const size_t mtry = 1){
    //  Fit regular DET on the trainingData
    arma::mat trainingData(train);
    DTree<arma::mat, int>* dtree = det::Trainer<arma::mat, int>(trainingData, folds, false, max_leaf_size, min_leaf_size, false, criterion, mtry);
    Log::Info << "training tree depth: " << dtree->SubtreeLeaves() << endl;
    
    //  Store the boundaries and density values of the hyperrectangles of the training DET
    std::vector<double> densities;
    std::vector<arma::Col<double>> upper;
    std::vector<arma::Col<double>> lower;
    GetLvlSets(dtree, densities, upper, lower);
    arma::Col<double> train_dens = arma::conv_to<arma::Col<double>>::from(densities);
    size_t n_lvls = upper.size();
    
    //  Compute the density values of validation dataset on the training DET
    arma::Col<double> val_dens(val.n_cols);
    for (size_t i = 0; i < val.n_cols; i++)
      val_dens(i) = dtree->ComputeValue(val.col(i));
    
    //  Initialize the conformal DET on the combined dataset
    arma::mat combinedData = arma::join_rows(train, val);
    arma::Col<size_t> oldFromNew(combinedData.n_cols);
    for (size_t i = 0; i < oldFromNew.n_elem; ++i)
      oldFromNew(i) = i;
    
    DTree<arma::mat, int>* ctree = new det::DTree<arma::mat, int>(combinedData, criterion);
    ctree->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
    
    //  If no need to grow the root node, return the tree
    if (ctree->GetReduction() <= 0){
        return ctree;
    }

    //  The mask variable indicates which hyperrectangles from the training DET intersect with the current node (root node)
    std::unordered_set<size_t> mask;
    for (size_t i = 0; i < n_lvls; ++i)
        mask.insert(i);
    ctree->SetMask(mask);
    
    //  Initialize the probability mass of the conformal tree on each hyperrectangle of the training DET
    arma::Col<double> ctree_mass(n_lvls);
    for (size_t i = 0; i < n_lvls; ++i){
        ctree_mass(i) = getMass(ctree, upper[i], lower[i]);
    }
    
    //  Initialize the priority queue for growing the conformal DET
    std::priority_queue<DTree<arma::mat, int>*, vector<DTree<arma::mat, int>*>, CompareTrees> pq;
    pq.push(ctree);
    
    double eps = 1e-10;
    arma::Col<double> inverse_val_dens = 1.0/(val_dens + eps);
    arma::Col<double> inverse_train_dens = 1.0/(train_dens + eps);
    arma::Col<double> unique_inverse_val_dens = arma::unique(inverse_val_dens);
    
    //  Generate the set of intervals on the density values
    std::vector<size_t> intv_ind_left;
    std::vector<size_t> intv_ind_right;
    genIntv(unique_inverse_val_dens.n_elem, intv_ind_left, intv_ind_right);
    size_t n_conds = intv_ind_left.size();
    
   //  The condMask variable indicates which set of conditions are related to the current node (root node) based on the set of hyperrectangles contained in the each condition
    std::unordered_set<size_t> condMask;
    for (size_t i = 0; i < n_conds; ++i)
        condMask.insert(i);
    ctree->SetCondMask(condMask);
    
    //  Generate the MLL conditions
    std::vector<double> Fvals;
    std::unordered_map<int, std::unordered_set<int>> umap;
    double kappa = msQuantile(delta);
    GenerateMLLConditions(inverse_train_dens,
                          inverse_val_dens,
                          unique_inverse_val_dens,
                          intv_ind_left,
                          intv_ind_right,
                          Fvals,
                          umap);
    
    // Keep track of the set of unsatisfied conditions
    std::unordered_set<int> unsatisfied_hash = {};
    Log::Info << "total conditions: " << n_conds << endl;
    bool curr_cond;
    double H;
    double F;
    for (int i : condMask){
        H = ProbMass(ctree_mass, umap[i]);
        F = Fvals[i];
        curr_cond = std::sqrt(2.0*logLR(val.n_cols, H, F)) <= LL(F) + kappa;
        if (!curr_cond){
            unsatisfied_hash.insert(i);
        }
    }
    
    // Keep track of the number of unsatisfied conditions
    int K = unsatisfied_hash.size();
    int temp;
    Log::Info << "initial satisfied conditions: " << n_conds - K << endl;
    
    DTree<arma::mat, int>* node;
    while (true){
        Log::Info << "tree leaves: " << ctree->Depth() << endl;
        temp = 0;
        node = pq.top();
        pq.pop();
        condMask = node->GetCondMask();
        Log::Info << "n_conds to check: " << condMask.size() << endl;
        
        // If the current node is doesn't affect the list of unsatisfied conditions, we skip the node. Otherwise, grow the node.
        bool ToGrow = false;
        for (int x : condMask){
            if (unsatisfied_hash.count(x) > 0){
                ToGrow = true;
                break;
            }
        }
        Log::Info << "To Grow: " << ToGrow << endl;
        
        if (ToGrow){
            // Grow the current node
            node->GrowOnce(combinedData, oldFromNew);
            
            // Update mask, condMask, and ctree_mass after the split
            UpdateTreeMass(node, ctree_mass, upper, lower, umap);
            
            // Update the set of unsatisfied conditions
            for (int i : condMask){
                H = ProbMass(ctree_mass, umap[i]);
                F = Fvals[i];
                curr_cond = std::sqrt(2.0*logLR(val.n_cols, H, F)) <= LL(F) + kappa;
                if (unsatisfied_hash.count(i) > 0 && curr_cond){
                    temp += 1;
                    unsatisfied_hash.erase(i);
                }
                else if (unsatisfied_hash.count(i) == 0 && !curr_cond){
                    temp -= 1;
                    unsatisfied_hash.insert(i);
                }
            }
            K -= temp;
            Log::Info << "remaining conditions: " << K << endl;
            assert(K == unsatisfied_hash.size());
            
            // if all conditions are satisfied, break out of the loop and return the tree
            if (K == 0){
                Log::Info << "conditions met" << endl;
                break;
            }
            
            // Add the left and right children of the current node to the priority queue
            (node->Left())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
            (node->Right())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
                                                                  
            if ((node->Left())->GetReduction() > 0){
                pq.push(node->Left());
            }
            if ((node->Right())->GetReduction() > 0){
                pq.push(node->Right());
            }
        }
        
        // If the priority queue is empty
        if (pq.size() == 0){
            Log::Info << "No Conformal Tree found." << endl;
            break;
        }
        
    }
    return ctree;
};

// DTree<arma::mat, int>* FitMLL(DTree<arma::mat, int>* train_tree,
//                               arma::mat &train,
//                               arma::mat &val,
//                               const double delta = 0.01,
//                               const std::string criterion = "NLL",
//                               const size_t max_leaf_size = 10,
//                               const size_t min_leaf_size = 5,
//                               const size_t folds = 10,
//                               const size_t mtry = 1){
//     //  Fit regular DET on the trainingData
//     // arma::mat trainingData(train);
//     DTree<arma::mat, int>* dtree = train_tree;
//     Log::Info << "training tree depth: " << dtree->SubtreeLeaves() << endl;
    
//     //  Store the boundaries and density values of the hyperrectangles of the training DET
//     std::vector<double> densities;
//     std::vector<arma::Col<double>> upper;
//     std::vector<arma::Col<double>> lower;
//     GetLvlSets(dtree, densities, upper, lower);
//     arma::Col<double> train_dens = arma::conv_to<arma::Col<double>>::from(densities);
//     size_t n_lvls = upper.size();
    
//     //  Compute the density values of validation dataset on the training DET
//     arma::Col<double> val_dens(val.n_cols);
//     for (size_t i = 0; i < val.n_cols; i++)
//       val_dens(i) = dtree->ComputeValue(val.col(i));
    
//     //  Initialize the conformal DET on the combined dataset
//     arma::mat combinedData = arma::join_rows(train, val);
//     arma::Col<size_t> oldFromNew(combinedData.n_cols);
//     for (size_t i = 0; i < oldFromNew.n_elem; ++i)
//       oldFromNew(i) = i;
    
//     DTree<arma::mat, int>* ctree = new det::DTree<arma::mat, int>(combinedData, criterion);
//     ctree->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
    
//     //  If no need to grow the root node, return the tree
//     if (ctree->GetReduction() <= 0){
//         return ctree;
//     }

//     //  The mask variable indicates which hyperrectangles from the training DET intersect with the current node (root node)
//     std::unordered_set<size_t> mask;
//     for (size_t i = 0; i < n_lvls; ++i)
//         mask.insert(i);
//     ctree->SetMask(mask);
    
//     //  Initialize the probability mass of the conformal tree on each hyperrectangle of the training DET
//     arma::Col<double> ctree_mass(n_lvls);
//     for (size_t i = 0; i < n_lvls; ++i){
//         ctree_mass(i) = getMass(ctree, upper[i], lower[i]);
//     }
    
//     //  Initialize the priority queue for growing the conformal DET
//     std::priority_queue<DTree<arma::mat, int>*, vector<DTree<arma::mat, int>*>, CompareTrees> pq;
//     pq.push(ctree);
    
//     double eps = 1e-10;
//     arma::Col<double> inverse_val_dens = 1.0/(val_dens + eps);
//     arma::Col<double> inverse_train_dens = 1.0/(train_dens + eps);
//     arma::Col<double> unique_inverse_val_dens = arma::unique(inverse_val_dens);
    
//     //  Generate the set of intervals on the density values
//     std::vector<size_t> intv_ind_left;
//     std::vector<size_t> intv_ind_right;
//     genIntv(unique_inverse_val_dens.n_elem, intv_ind_left, intv_ind_right);
//     size_t n_conds = intv_ind_left.size();
    
//    //  The condMask variable indicates which set of conditions are related to the current node (root node) based on the set of hyperrectangles contained in the each condition
//     std::unordered_set<size_t> condMask;
//     for (size_t i = 0; i < n_conds; ++i)
//         condMask.insert(i);
//     ctree->SetCondMask(condMask);
    
//     //  Generate the MLL conditions
//     std::vector<double> Fvals;
//     std::unordered_map<int, std::unordered_set<int>> umap;
//     double kappa = msQuantile(delta);
//     GenerateMLLConditions(inverse_train_dens,
//                           inverse_val_dens,
//                           unique_inverse_val_dens,
//                           intv_ind_left,
//                           intv_ind_right,
//                           Fvals,
//                           umap);
    
//     // Keep track of the set of unsatisfied conditions
//     std::unordered_set<int> unsatisfied_hash = {};
//     Log::Info << "total conditions: " << n_conds << endl;
//     bool curr_cond;
//     double H;
//     double F;
//     for (int i : condMask){
//         H = ProbMass(ctree_mass, umap[i]);
//         F = Fvals[i];
//         curr_cond = std::sqrt(2.0*logLR(val.n_cols, H, F)) <= LL(F) + kappa;
//         if (!curr_cond){
//             unsatisfied_hash.insert(i);
//         }
//     }
    
//     // Keep track of the number of unsatisfied conditions
//     int K = unsatisfied_hash.size();
//     int temp;
//     Log::Info << "initial satisfied conditions: " << n_conds - K << endl;
    
//     DTree<arma::mat, int>* node;
//     while (true){
//         Log::Info << "tree leaves: " << ctree->Depth() << endl;
//         temp = 0;
//         node = pq.top();
//         pq.pop();
//         condMask = node->GetCondMask();
//         Log::Info << "n_conds to check: " << condMask.size() << endl;
        
//         // If the current node is doesn't affect the list of unsatisfied conditions, we skip the node. Otherwise, grow the node.
//         bool ToGrow = false;
//         for (int x : condMask){
//             if (unsatisfied_hash.count(x) > 0){
//                 ToGrow = true;
//                 break;
//             }
//         }
//         Log::Info << "To Grow: " << ToGrow << endl;
        
//         if (ToGrow){
//             // Grow the current node
//             node->GrowOnce(combinedData, oldFromNew);
            
//             // Update mask, condMask, and ctree_mass after the split
//             UpdateTreeMass(node, ctree_mass, upper, lower, umap);
            
//             // Update the set of unsatisfied conditions
//             for (int i : condMask){
//                 H = ProbMass(ctree_mass, umap[i]);
//                 F = Fvals[i];
//                 curr_cond = std::sqrt(2.0*logLR(val.n_cols, H, F)) <= LL(F) + kappa;
//                 if (unsatisfied_hash.count(i) > 0 && curr_cond){
//                     temp += 1;
//                     unsatisfied_hash.erase(i);
//                 }
//                 else if (unsatisfied_hash.count(i) == 0 && !curr_cond){
//                     temp -= 1;
//                     unsatisfied_hash.insert(i);
//                 }
//             }
//             K -= temp;
//             Log::Info << "remaining conditions: " << K << endl;
//             assert(K == unsatisfied_hash.size());
            
//             // if all conditions are satisfied, break out of the loop and return the tree
//             if (K == 0){
//                 Log::Info << "conditions met" << endl;
//                 break;
//             }
            
//             // Add the left and right children of the current node to the priority queue
//             (node->Left())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
//             (node->Right())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
                                                                  
//             if ((node->Left())->GetReduction() > 0){
//                 pq.push(node->Left());
//             }
//             if ((node->Right())->GetReduction() > 0){
//                 pq.push(node->Right());
//             }
//         }
        
//         // If the priority queue is empty
//         if (pq.size() == 0){
//             Log::Info << "No Conformal Tree found." << endl;
//             break;
//         }
        
//     }
//     return ctree;
// };

void GenerateVCConditions(double delta,
                          arma::Col<double>& train_dens,
                          arma::Col<double>& val_dens,
                          arma::Col<double>& lvl_list,
                          std::vector<double>& lower_bound,
                          std::vector<double>& upper_bound,
                          std::unordered_map<int, std::unordered_set<int>>& umap){
    size_t N_val = val_dens.n_elem;
    size_t N_lvl = lvl_list.n_elem;
    double beta = std::sqrt((4.0*std::log(2*N_val*std::exp(1)) + 4.0*std::log(4/delta))/N_val);
    
    double p_hat;
    arma::uvec inds;
    std::unordered_set<int> uset;
    double lvl;
    for (size_t i = 0; i < N_lvl; ++i){
        lvl = lvl_list(i);
        p_hat = (double) ((arma::uvec) arma::find(val_dens >= lvl)).n_elem/N_val;
        upper_bound.push_back(std::min(p_hat + std::pow(beta,2)/2.0 + beta*std::sqrt(p_hat + std::pow(beta,2)/4.0), 1.0));
        if (p_hat < 3.0*std::pow(beta,2)/4.0){
            lower_bound.push_back(std::max(0.0, p_hat - beta*std::sqrt(p_hat)));
        }
        else{
            lower_bound.push_back(std::max(0.0, std::max(p_hat - beta*std::sqrt(p_hat), p_hat - std::pow(beta,2)/2.0 - beta*std::sqrt(p_hat - 3.0*std::pow(beta,2)/4.0))));
        }
        
        inds = arma::find(train_dens >= lvl);
        uset = {};
        uset.insert(inds.begin(), inds.end());
        umap.insert({i, uset});
    }
};

void GenerateEPConditions(double xi,
                          arma::Col<double>& train_dens,
                          arma::Col<double>& val_dens,
                          arma::Col<double>& lvl_list,
                          std::vector<double>& lower_bound,
                          std::vector<double>& upper_bound,
                          std::unordered_map<int, std::unordered_set<int>>& umap){
    size_t N_val = val_dens.n_elem;
    size_t N_lvl = lvl_list.n_elem;
    
    double p_hat;
    arma::uvec inds;
    std::unordered_set<int> uset;
    double lvl;
    for (size_t i = 0; i < N_lvl; ++i){
        lvl = lvl_list(i);
        p_hat = (double) ((arma::uvec) arma::find(val_dens >= lvl)).n_elem/N_val;
        upper_bound.push_back((p_hat + std::pow(xi,2)/(2*N_val) + std::sqrt(std::pow(xi,4)/(4.0*std::pow((double) N_val,2)) + p_hat*(1-p_hat)*std::pow(xi,2)/N_val))/(1.0 + std::pow(xi,2)/N_val));
        lower_bound.push_back((p_hat + std::pow(xi,2)/(2*N_val) - std::sqrt(std::pow(xi,4)/(4.0*std::pow((double) N_val,2)) + p_hat*(1-p_hat)*std::pow(xi,2)/N_val))/(1.0 + std::pow(xi,2)/N_val));
        
        inds = arma::find(train_dens >= lvl);
        uset = {};
        uset.insert(inds.begin(), inds.end());
        umap.insert({i, uset});
    }
};


DTree<arma::mat, int>* FitVC(arma::mat &train,
                             arma::mat &val,
                             const double delta = 0.01,
                             const std::string criterion = "NLL",
                             const size_t max_leaf_size = 10,
                             const size_t min_leaf_size = 5,
                             const size_t folds = 10,
                             const size_t mtry = 1){
    
    //  Fit regular DET on the trainingData
    arma::mat trainingData(train);
    DTree<arma::mat, int>* dtree = det::Trainer<arma::mat, int>(trainingData, folds, false, max_leaf_size, min_leaf_size, false, criterion, mtry);
    Log::Info << "training tree depth: " << dtree->SubtreeLeaves() << endl;
    
    //  Store the boundaries and density values of the hyperrectangles of the training DET
    std::vector<double> densities;
    std::vector<arma::Col<double>> upper;
    std::vector<arma::Col<double>> lower;
    GetLvlSets(dtree, densities, upper, lower);
    arma::Col<double> train_dens = arma::conv_to<arma::Col<double>>::from(densities);
    size_t n_lvls = upper.size();
    
    arma::Col<double> lvl_list = arma::unique(train_dens);
    size_t n_conds = lvl_list.n_elem;
    
    //  Compute the density values of validation dataset on the training DET
    arma::Col<double> val_dens(val.n_cols);
    for (size_t i = 0; i < val.n_cols; i++)
      val_dens(i) = dtree->ComputeValue(val.col(i));
    
    //  Initialize the conformal DET on the combined dataset
    arma::mat combinedData = arma::join_rows(train, val);
    arma::Col<size_t> oldFromNew(combinedData.n_cols);
    for (size_t i = 0; i < oldFromNew.n_elem; ++i)
      oldFromNew(i) = i;
    
    DTree<arma::mat, int>* ctree = new det::DTree<arma::mat, int>(combinedData, criterion);
    ctree->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
    
    //  If no need to grow the root node, return the tree
    if (ctree->GetReduction() <= 0){
        return ctree;
    }

    //  The mask variable indicates which hyperrectangles from the training DET intersect with the current node (root node)
    std::unordered_set<size_t> mask;
    for (size_t i = 0; i < n_lvls; ++i)
        mask.insert(i);
    ctree->SetMask(mask);
    
    //  Initialize the probability mass of the conformal tree on each hyperrectangle of the training DET
    arma::Col<double> ctree_mass(n_lvls);
    for (size_t i = 0; i < n_lvls; ++i){
        ctree_mass(i) = getMass(ctree, upper[i], lower[i]);
    }
    
    //  Initialize the priority queue for growing the conformal DET
    std::priority_queue<DTree<arma::mat, int>*, vector<DTree<arma::mat, int>*>, CompareTrees> pq;
    pq.push(ctree);
    
   //  The condMask variable indicates which set of conditions are related to the current node (root node) based on the set of hyperrectangles contained in the each condition
    std::unordered_set<size_t> condMask;
    for (size_t i = 0; i < n_conds; ++i)
        condMask.insert(i);
    ctree->SetCondMask(condMask);
    
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
    std::unordered_map<int, std::unordered_set<int>> umap;
    GenerateVCConditions(delta, train_dens, val_dens, lvl_list, lower_bound, upper_bound, umap);

    // Keep track of the set of unsatisfied conditions
    std::unordered_set<int> unsatisfied_hash = {};
    Log::Info << "total conditions: " << n_conds << endl;
    bool curr_cond;
    double p_cur;
    for (int i : condMask){
        p_cur = ProbMass(ctree_mass, umap[i]);
        curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
        if (!curr_cond){
            unsatisfied_hash.insert(i);
        }
    }
    
    // Keep track of the number of unsatisfied conditions
    int K = unsatisfied_hash.size();
    int temp;
    Log::Info << "initial satisfied conditions: " << n_conds - K << endl;
    
    DTree<arma::mat, int>* node;
    while (true){
        Log::Info << "tree leaves: " << ctree->Depth() << endl;
        temp = 0;
        node = pq.top();
        pq.pop();
        condMask = node->GetCondMask();
        Log::Info << "n_conds to check: " << condMask.size() << endl;
        
        // If the current node is doesn't affect the list of unsatisfied conditions, we skip the node. Otherwise, grow the node.
        bool ToGrow = false;
        for (int x : condMask){
            if (unsatisfied_hash.count(x) > 0){
                ToGrow = true;
                break;
            }
        }
        Log::Info << "To Grow: " << ToGrow << endl;
        
        if (ToGrow){
            // Grow the current node
            node->GrowOnce(combinedData, oldFromNew);
            
            // Update mask, condMask, and ctree_mass after the split
            UpdateTreeMass(node, ctree_mass, upper, lower, umap);
            
            // Update the set of unsatisfied conditions
            for (int i : condMask){
                p_cur = ProbMass(ctree_mass, umap[i]);
                curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
                if (unsatisfied_hash.count(i) > 0 && curr_cond){
                    temp += 1;
                    unsatisfied_hash.erase(i);
                }
                else if (unsatisfied_hash.count(i) == 0 && !curr_cond){
                    temp -= 1;
                    unsatisfied_hash.insert(i);
                }
            }
            K -= temp;
            Log::Info << "remaining conditions: " << K << endl;
            assert(K == unsatisfied_hash.size());
            
            // if all conditions are satisfied, break out of the loop and return the tree
            if (K == 0){
                Log::Info << "conditions met" << endl;
                break;
            }
            
            // Add the left and right children of the current node to the priority queue
            (node->Left())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
            (node->Right())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
                                                                  
            if ((node->Left())->GetReduction() > 0){
                pq.push(node->Left());
            }
            if ((node->Right())->GetReduction() > 0){
                pq.push(node->Right());
            }
        }
        
        // If the priority queue is empty
        if (pq.size() == 0){
            Log::Info << "No Conformal Tree found." << endl;
            break;
        }
        
    }
    return ctree;
};

// DTree<arma::mat, int>* FitVC(DTree<arma::mat, int>* train_tree,
//                              arma::mat &train,
//                              arma::mat &val,
//                              const double delta = 0.01,
//                              const std::string criterion = "NLL",
//                              const size_t max_leaf_size = 10,
//                              const size_t min_leaf_size = 5,
//                              const size_t folds = 10,
//                              const size_t mtry = 1){
    
//     //  Fit regular DET on the trainingData
//     // arma::mat trainingData(train);
//     DTree<arma::mat, int>* dtree = train_tree;
//     Log::Info << "training tree depth: " << dtree->SubtreeLeaves() << endl;
    
//     //  Store the boundaries and density values of the hyperrectangles of the training DET
//     std::vector<double> densities;
//     std::vector<arma::Col<double>> upper;
//     std::vector<arma::Col<double>> lower;
//     GetLvlSets(dtree, densities, upper, lower);
//     arma::Col<double> train_dens = arma::conv_to<arma::Col<double>>::from(densities);
//     size_t n_lvls = upper.size();
    
//     arma::Col<double> lvl_list = arma::unique(train_dens);
//     size_t n_conds = lvl_list.n_elem;
    
//     //  Compute the density values of validation dataset on the training DET
//     arma::Col<double> val_dens(val.n_cols);
//     for (size_t i = 0; i < val.n_cols; i++)
//       val_dens(i) = dtree->ComputeValue(val.col(i));
    
//     //  Initialize the conformal DET on the combined dataset
//     arma::mat combinedData = arma::join_rows(train, val);
//     arma::Col<size_t> oldFromNew(combinedData.n_cols);
//     for (size_t i = 0; i < oldFromNew.n_elem; ++i)
//       oldFromNew(i) = i;
    
//     DTree<arma::mat, int>* ctree = new det::DTree<arma::mat, int>(combinedData, criterion);
//     ctree->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
    
//     //  If no need to grow the root node, return the tree
//     if (ctree->GetReduction() <= 0){
//         return ctree;
//     }

//     //  The mask variable indicates which hyperrectangles from the training DET intersect with the current node (root node)
//     std::unordered_set<size_t> mask;
//     for (size_t i = 0; i < n_lvls; ++i)
//         mask.insert(i);
//     ctree->SetMask(mask);
    
//     //  Initialize the probability mass of the conformal tree on each hyperrectangle of the training DET
//     arma::Col<double> ctree_mass(n_lvls);
//     for (size_t i = 0; i < n_lvls; ++i){
//         ctree_mass(i) = getMass(ctree, upper[i], lower[i]);
//     }
    
//     //  Initialize the priority queue for growing the conformal DET
//     std::priority_queue<DTree<arma::mat, int>*, vector<DTree<arma::mat, int>*>, CompareTrees> pq;
//     pq.push(ctree);
    
//    //  The condMask variable indicates which set of conditions are related to the current node (root node) based on the set of hyperrectangles contained in the each condition
//     std::unordered_set<size_t> condMask;
//     for (size_t i = 0; i < n_conds; ++i)
//         condMask.insert(i);
//     ctree->SetCondMask(condMask);
    
//     std::vector<double> lower_bound;
//     std::vector<double> upper_bound;
//     std::unordered_map<int, std::unordered_set<int>> umap;
//     GenerateVCConditions(delta, train_dens, val_dens, lvl_list, lower_bound, upper_bound, umap);

//     // Keep track of the set of unsatisfied conditions
//     std::unordered_set<int> unsatisfied_hash = {};
//     Log::Info << "total conditions: " << n_conds << endl;
//     bool curr_cond;
//     double p_cur;
//     for (int i : condMask){
//         p_cur = ProbMass(ctree_mass, umap[i]);
//         curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
//         if (!curr_cond){
//             unsatisfied_hash.insert(i);
//         }
//     }
    
//     // Keep track of the number of unsatisfied conditions
//     int K = unsatisfied_hash.size();
//     int temp;
//     Log::Info << "initial satisfied conditions: " << n_conds - K << endl;
    
//     DTree<arma::mat, int>* node;
//     while (true){
//         Log::Info << "tree leaves: " << ctree->Depth() << endl;
//         temp = 0;
//         node = pq.top();
//         pq.pop();
//         condMask = node->GetCondMask();
//         Log::Info << "n_conds to check: " << condMask.size() << endl;
        
//         // If the current node is doesn't affect the list of unsatisfied conditions, we skip the node. Otherwise, grow the node.
//         bool ToGrow = false;
//         for (int x : condMask){
//             if (unsatisfied_hash.count(x) > 0){
//                 ToGrow = true;
//                 break;
//             }
//         }
//         Log::Info << "To Grow: " << ToGrow << endl;
        
//         if (ToGrow){
//             // Grow the current node
//             node->GrowOnce(combinedData, oldFromNew);
            
//             // Update mask, condMask, and ctree_mass after the split
//             UpdateTreeMass(node, ctree_mass, upper, lower, umap);
            
//             // Update the set of unsatisfied conditions
//             for (int i : condMask){
//                 p_cur = ProbMass(ctree_mass, umap[i]);
//                 curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
//                 if (unsatisfied_hash.count(i) > 0 && curr_cond){
//                     temp += 1;
//                     unsatisfied_hash.erase(i);
//                 }
//                 else if (unsatisfied_hash.count(i) == 0 && !curr_cond){
//                     temp -= 1;
//                     unsatisfied_hash.insert(i);
//                 }
//             }
//             K -= temp;
//             Log::Info << "remaining conditions: " << K << endl;
//             assert(K == unsatisfied_hash.size());
            
//             // if all conditions are satisfied, break out of the loop and return the tree
//             if (K == 0){
//                 Log::Info << "conditions met" << endl;
//                 break;
//             }
            
//             // Add the left and right children of the current node to the priority queue
//             (node->Left())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
//             (node->Right())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
                                                                  
//             if ((node->Left())->GetReduction() > 0){
//                 pq.push(node->Left());
//             }
//             if ((node->Right())->GetReduction() > 0){
//                 pq.push(node->Right());
//             }
//         }
        
//         // If the priority queue is empty
//         if (pq.size() == 0){
//             Log::Info << "No Conformal Tree found." << endl;
//             break;
//         }
        
//     }
//     return ctree;
// };

DTree<arma::mat, int>* FitEP(arma::mat &train,
                             arma::mat &val,
                             const double delta = 0.01,
                             const std::string criterion = "NLL",
                             const size_t max_leaf_size = 10,
                             const size_t min_leaf_size = 5,
                             const size_t folds = 10,
                             const size_t mtry = 1,
                             std::string typ = "sym"){

    //  Fit regular DET on the trainingData
    arma::mat trainingData(train);
    DTree<arma::mat, int>* dtree = det::Trainer<arma::mat, int>(trainingData, folds, false, max_leaf_size, min_leaf_size, false, criterion, mtry);
    Log::Info << "training tree depth: " << dtree->SubtreeLeaves() << endl;
    
    //  Store the boundaries and density values of the hyperrectangles of the training DET
    std::vector<double> densities;
    std::vector<arma::Col<double>> upper;
    std::vector<arma::Col<double>> lower;
    GetLvlSets(dtree, densities, upper, lower);
    arma::Col<double> train_dens = arma::conv_to<arma::Col<double>>::from(densities);
    size_t n_lvls = upper.size();
    
    arma::Col<double> lvl_list = arma::unique(train_dens);
    size_t n_conds = lvl_list.n_elem;

    double xi = GenerateXi(val.n_cols, 10000, delta, delta, typ);
    
    //  Compute the density values of validation dataset on the training DET
    arma::Col<double> val_dens(val.n_cols);
    for (size_t i = 0; i < val.n_cols; i++)
      val_dens(i) = dtree->ComputeValue(val.col(i));
    
    //  Initialize the conformal DET on the combined dataset
    arma::mat combinedData = arma::join_rows(train, val);
    arma::Col<size_t> oldFromNew(combinedData.n_cols);
    for (size_t i = 0; i < oldFromNew.n_elem; ++i)
      oldFromNew(i) = i;
    
    DTree<arma::mat, int>* ctree = new det::DTree<arma::mat, int>(combinedData, criterion);
    ctree->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
    
    //  If no need to grow the root node, return the tree
    if (ctree->GetReduction() <= 0){
        return ctree;
    }

    //  The mask variable indicates which hyperrectangles from the training DET intersect with the current node (root node)
    std::unordered_set<size_t> mask;
    for (size_t i = 0; i < n_lvls; ++i)
        mask.insert(i);
    ctree->SetMask(mask);
    
    //  Initialize the probability mass of the conformal tree on each hyperrectangle of the training DET
    arma::Col<double> ctree_mass(n_lvls);
    for (size_t i = 0; i < n_lvls; ++i){
        ctree_mass(i) = getMass(ctree, upper[i], lower[i]);
    }
    
    //  Initialize the priority queue for growing the conformal DET
    std::priority_queue<DTree<arma::mat, int>*, vector<DTree<arma::mat, int>*>, CompareTrees> pq;
    pq.push(ctree);
    
   //  The condMask variable indicates which set of conditions are related to the current node (root node) based on the set of hyperrectangles contained in the each condition
    std::unordered_set<size_t> condMask;
    for (size_t i = 0; i < n_conds; ++i)
        condMask.insert(i);
    ctree->SetCondMask(condMask);
    
    std::vector<double> lower_bound;
    std::vector<double> upper_bound;
    std::unordered_map<int, std::unordered_set<int>> umap;
    GenerateEPConditions(xi, train_dens, val_dens, lvl_list, lower_bound, upper_bound, umap);

    // Keep track of the set of unsatisfied conditions
    std::unordered_set<int> unsatisfied_hash = {};
    Log::Info << "total conditions: " << n_conds << endl;
    bool curr_cond;
    double p_cur;
    for (int i : condMask){
        p_cur = ProbMass(ctree_mass, umap[i]);
        curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
        if (!curr_cond){
            unsatisfied_hash.insert(i);
        }
    }
    
    // Keep track of the number of unsatisfied conditions
    int K = unsatisfied_hash.size();
    int temp;
    Log::Info << "initial satisfied conditions: " << n_conds - K << endl;
    
    DTree<arma::mat, int>* node;
    while (true){
        Log::Info << "tree leaves: " << ctree->Depth() << endl;
        temp = 0;
        node = pq.top();
        pq.pop();
        condMask = node->GetCondMask();
        Log::Info << "n_conds to check: " << condMask.size() << endl;
        
        // If the current node is doesn't affect the list of unsatisfied conditions, we skip the node. Otherwise, grow the node.
        bool ToGrow = false;
        for (int x : condMask){
            if (unsatisfied_hash.count(x) > 0){
                ToGrow = true;
                break;
            }
        }
        Log::Info << "To Grow: " << ToGrow << endl;
        
        if (ToGrow){
            // Grow the current node
            node->GrowOnce(combinedData, oldFromNew);
            
            // Update mask, condMask, and ctree_mass after the split
            UpdateTreeMass(node, ctree_mass, upper, lower, umap);
            
            // Update the set of unsatisfied conditions
            for (int i : condMask){
                p_cur = ProbMass(ctree_mass, umap[i]);
                curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
                if (unsatisfied_hash.count(i) > 0 && curr_cond){
                    temp += 1;
                    unsatisfied_hash.erase(i);
                }
                else if (unsatisfied_hash.count(i) == 0 && !curr_cond){
                    temp -= 1;
                    unsatisfied_hash.insert(i);
                }
            }
            K -= temp;
            Log::Info << "remaining conditions: " << K << endl;
            assert(K == unsatisfied_hash.size());
            
            // if all conditions are satisfied, break out of the loop and return the tree
            if (K == 0){
                Log::Info << "conditions met" << endl;
                break;
            }
            
            // Add the left and right children of the current node to the priority queue
            (node->Left())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
            (node->Right())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
                                                                  
            if ((node->Left())->GetReduction() > 0){
                pq.push(node->Left());
            }
            if ((node->Right())->GetReduction() > 0){
                pq.push(node->Right());
            }
        }
        
        // If the priority queue is empty
        if (pq.size() == 0){
            Log::Info << "No Conformal Tree found." << endl;
            break;
        }
        
    }
    return ctree;
};

// DTree<arma::mat, int>* FitEP(DTree<arma::mat, int>* train_tree,
//                              arma::mat &train, 
//                              arma::mat &val,
//                              const double delta = 0.01,
//                              const std::string criterion = "NLL",
//                              const size_t max_leaf_size = 10,
//                              const size_t min_leaf_size = 5,
//                              const size_t folds = 10,
//                              const size_t mtry = 1,
//                              std::string typ = "sym"){
    
//     double xi = GenerateXi(val.n_cols, 10000, delta, delta, typ);

//     //  Fit regular DET on the trainingData
//     DTree<arma::mat, int>* dtree = train_tree;
//     Log::Info << "training tree depth: " << dtree->SubtreeLeaves() << endl;
    
//     //  Store the boundaries and density values of the hyperrectangles of the training DET
//     std::vector<double> densities;
//     std::vector<arma::Col<double>> upper;
//     std::vector<arma::Col<double>> lower;
//     GetLvlSets(dtree, densities, upper, lower);
//     arma::Col<double> train_dens = arma::conv_to<arma::Col<double>>::from(densities);
//     size_t n_lvls = upper.size();
    
//     arma::Col<double> lvl_list = arma::unique(train_dens);
//     size_t n_conds = lvl_list.n_elem;
    
//     //  Compute the density values of validation dataset on the training DET
//     arma::Col<double> val_dens(val.n_cols);
//     for (size_t i = 0; i < val.n_cols; i++)
//       val_dens(i) = dtree->ComputeValue(val.col(i));
    
//     //  Initialize the conformal DET on the combined dataset
//     arma::mat combinedData = arma::join_rows(train, val);
//     arma::Col<size_t> oldFromNew(combinedData.n_cols);
//     for (size_t i = 0; i < oldFromNew.n_elem; ++i)
//       oldFromNew(i) = i;
    
//     DTree<arma::mat, int>* ctree = new det::DTree<arma::mat, int>(combinedData, criterion);
//     ctree->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
    
//     //  If no need to grow the root node, return the tree
//     if (ctree->GetReduction() <= 0){
//         return ctree;
//     }

//     //  The mask variable indicates which hyperrectangles from the training DET intersect with the current node (root node)
//     std::unordered_set<size_t> mask;
//     for (size_t i = 0; i < n_lvls; ++i)
//         mask.insert(i);
//     ctree->SetMask(mask);
    
//     //  Initialize the probability mass of the conformal tree on each hyperrectangle of the training DET
//     arma::Col<double> ctree_mass(n_lvls);
//     for (size_t i = 0; i < n_lvls; ++i){
//         ctree_mass(i) = getMass(ctree, upper[i], lower[i]);
//     }
    
//     //  Initialize the priority queue for growing the conformal DET
//     std::priority_queue<DTree<arma::mat, int>*, vector<DTree<arma::mat, int>*>, CompareTrees> pq;
//     pq.push(ctree);
    
//    //  The condMask variable indicates which set of conditions are related to the current node (root node) based on the set of hyperrectangles contained in the each condition
//     std::unordered_set<size_t> condMask;
//     for (size_t i = 0; i < n_conds; ++i)
//         condMask.insert(i);
//     ctree->SetCondMask(condMask);
    
//     std::vector<double> lower_bound;
//     std::vector<double> upper_bound;
//     std::unordered_map<int, std::unordered_set<int>> umap;
//     GenerateEPConditions(xi, train_dens, val_dens, lvl_list, lower_bound, upper_bound, umap);

//     // Keep track of the set of unsatisfied conditions
//     std::unordered_set<int> unsatisfied_hash = {};
//     Log::Info << "total conditions: " << n_conds << endl;
//     bool curr_cond;
//     double p_cur;
//     for (int i : condMask){
//         p_cur = ProbMass(ctree_mass, umap[i]);
//         curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
//         if (!curr_cond){
//             unsatisfied_hash.insert(i);
//         }
//     }
    
//     // Keep track of the number of unsatisfied conditions
//     int K = unsatisfied_hash.size();
//     int temp;
//     Log::Info << "initial satisfied conditions: " << n_conds - K << endl;
    
//     DTree<arma::mat, int>* node;
//     while (true){
//         Log::Info << "tree leaves: " << ctree->Depth() << endl;
//         temp = 0;
//         node = pq.top();
//         pq.pop();
//         condMask = node->GetCondMask();
//         Log::Info << "n_conds to check: " << condMask.size() << endl;
        
//         // If the current node is doesn't affect the list of unsatisfied conditions, we skip the node. Otherwise, grow the node.
//         bool ToGrow = false;
//         for (int x : condMask){
//             if (unsatisfied_hash.count(x) > 0){
//                 ToGrow = true;
//                 break;
//             }
//         }
//         Log::Info << "To Grow: " << ToGrow << endl;
        
//         if (ToGrow){
//             // Grow the current node
//             node->GrowOnce(combinedData, oldFromNew);
            
//             // Update mask, condMask, and ctree_mass after the split
//             UpdateTreeMass(node, ctree_mass, upper, lower, umap);
            
//             // Update the set of unsatisfied conditions
//             for (int i : condMask){
//                 p_cur = ProbMass(ctree_mass, umap[i]);
//                 curr_cond = ((lower_bound[i] <= p_cur) && (p_cur <= upper_bound[i]));
//                 if (unsatisfied_hash.count(i) > 0 && curr_cond){
//                     temp += 1;
//                     unsatisfied_hash.erase(i);
//                 }
//                 else if (unsatisfied_hash.count(i) == 0 && !curr_cond){
//                     temp -= 1;
//                     unsatisfied_hash.insert(i);
//                 }
//             }
//             K -= temp;
//             Log::Info << "remaining conditions: " << K << endl;
//             assert(K == unsatisfied_hash.size());
            
//             // if all conditions are satisfied, break out of the loop and return the tree
//             if (K == 0){
//                 Log::Info << "conditions met" << endl;
//                 break;
//             }
            
//             // Add the left and right children of the current node to the priority queue
//             (node->Left())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
//             (node->Right())->ComputeLossReduction(combinedData, max_leaf_size, min_leaf_size, mtry);
                                                                  
//             if ((node->Left())->GetReduction() > 0){
//                 pq.push(node->Left());
//             }
//             if ((node->Right())->GetReduction() > 0){
//                 pq.push(node->Right());
//             }
//         }
        
//         // If the priority queue is empty
//         if (pq.size() == 0){
//             Log::Info << "No Conformal Tree found." << endl;
//             break;
//         }
        
//     }
//     return ctree;
// };

DTree<arma::mat, int>* FitConformal(arma::mat &train,
                                    arma::mat &val,
                                    std::string method,
                                    const double delta = 0.01,
                                    const std::string criterion = "NLL",
                                    const size_t max_leaf_size = 10,
                                    const size_t min_leaf_size = 5,
                                    const size_t folds = 10,
                                    const size_t mtry = 1,
                                    const std::string typ = "sym"){
    if (method == "mll"){
        return FitMLL(train, val, delta, criterion, max_leaf_size, min_leaf_size, folds, mtry);
    }
    else if (method == "vc"){
        return FitVC(train, val, delta, criterion, max_leaf_size, min_leaf_size, folds, mtry);
    }
    else{
        return FitEP(train, val, delta, criterion, max_leaf_size, min_leaf_size, folds, mtry, typ);
    }
};

// DTree<arma::mat, int>* FitConformal(DTree<arma::mat, int>* train_tree,
//                                     arma::mat &train,
//                                     arma::mat &val,
//                                     std::string method,
//                                     const double delta = 0.01,
//                                     const std::string criterion = "NLL",
//                                     const size_t max_leaf_size = 10,
//                                     const size_t min_leaf_size = 5,
//                                     const size_t folds = 10,
//                                     const size_t mtry = 1,
//                                     const std::string typ = "sym"){
//     if (method == "mll"){
//         return FitMLL(train_tree, train, val, delta, criterion, max_leaf_size, min_leaf_size, folds, mtry);
//     }
//     else if (method == "vc"){
//         return FitVC(train_tree, train, val, delta, criterion, max_leaf_size, min_leaf_size, folds, mtry);
//     }
//     else{
//         return FitEP(train_tree, train, val, delta, criterion, max_leaf_size, min_leaf_size, folds, mtry, typ);
//     }
// };

}
}
#endif /* conf_dTree_utils_hpp */

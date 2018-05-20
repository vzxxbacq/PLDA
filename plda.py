import scipy
import numpy as np
import math


M_LOG_2PI = 1.8378770664093454835606594728112


class Pldaconfig(object):
    def __init__(self, normalize_length=True, simple_length_norm=False):
        self.normalize_length = normalize_length
        self.simple_length_norm = simple_length_norm


class ClassInfo(object):
    def __init__(self, weight=0, num_example=0, mean=0):
        self.weight = weight
        self.num_example = num_example
        self.mean = mean


class PldaStats(object):
    def __init__(self, dim):
        self.dim_ = dim
        self.num_example = 0
        self.num_classes = 0
        self.class_weight = 0
        self.example_weight = 0
        self.sum = np.zeros(dim)
        self.offset_scatter= np.zeros([dim, dim])
        self.classinfo = list()

    def add_samples(self, weight, group):
        
        # Each row represent an utts of the same speaker.
        n = group.shape[-1]
        mean = np.mean(group, axis=0)
        
        self.offset_scatter += weight * group.T * group
        self.offset_scatter += -n * weight * mean * mean.T
        
        self.classinfo.append(ClassInfo(weight, n, mean))

        self.num_example += n
        self.num_classes += 1
        self.class_weight += weight
        self.example_weight += weight * n
        self.sum += weight * mean

    @property
    def is_sorted(self):
        for i in range(self.num_classes):
            if self.classinfo[i].num_example <= self.classinfo[i].num_classes:
                return False
        return True

    def sort(self):
        for i in range(self.num_classes) - 1:
            j = i
            while j < range(self.num_classes) - 1:
                if self.classinfo[j].num_example < self.classinfo[j+1].num_example:
                    tmp = self.classinfo[j]
                    self.classinfo[j] = self.classinfo[j+1]
                    self.classinfo[j+1] = tmp
        return


class PLDA(object):
    def __init__(self):
        self.mean = 0
        self.transform = 0
        self.psi = 0
        self.offset = 0
        self.dim = 0

    def transform_ivector(self, config, ivector, num_example):
        self.dim = ivector.shape[-1]
        transformed_ivec = self.offset
        transformed_ivec = 1.0 * self.transform * ivector + 1.0 * self.transform
        if(config.simple_length_norm):
            normalization_factor = math.sqrt(self.dim) / np.linalg.norm(transformed_ivec)
        else:
            normalization_factor = self.get_normalization_factor(transformed_ivec,
                                                            num_example)
        if(config.normalize_length):
            transformed_ivec = normalization_factor * transformed_ivec
        return transformed_ivec

    def log_likelihood_ratio(self, transform_train_ivector, num_utts, 
        transform_test_ivector):
        self.dim = transform_train_ivector.shape[-1]
        mean = np.zeros(self.dim)
        variance = np.zeros(self.dim)
        for i in range(self.dim):
            mean[i] = num_utts * self.psi[i] / (num_utts * self.psi[i] + 1.0)*transform_train_ivector[i]
            variance[i] = 1.0 + self.psi[i] / (num_utts * self.psi[i] + 1.0)
        #
        logdet = np.sum(np.log(variance))
        sqdiff = transform_test_ivector - mean
        sqdiff = np.power(sqdiff, np.full(sqdiff.shape, 2.0))
        variance = np.reciprocal(variance)
        loglike_given_class = -0.5 * (logdet + M_LOG_2PI * self.dim + np.dot(sqdiff, variance))
        #
        sqdiff = transform_test_ivector
        sqdiff = np.power(sqdiff, np.full(sqdiff.shape, 2.0))
        variance = self.psi + 1.0
        logdet = np.sum(np.log(variance))
        variance = np.reciprocal(variance)
        loglike_without_class = -0.5 * (logdet + M_LOG_2PI * self.dim + np.dot(sqdiff, variance))
        loglike_ratio = loglike_given_class - loglike_without_class
        return loglike_ratio

    def smooth_within_class_covariance(self, smoothing_factor):
        within_class_covar = np.ones(self.dim)
        smooth = np.full(self.dim,
                        smoothing_factor*within_class_covar*self.psi.T)
        within_class_covar = np.add(within_class_covar,
                                    smooth)
        self.psi = np.divide(self.psi, within_class_covar)
        within_class_covar = np.power(within_class_covar,
                                    np.full(within_class_covar.shape, -0.5))
        self.transform = np.diag(within_class_covar) * self.transform
        self.compute_derived_vars()

    # The method which needn't to use EM algorithm and calc the closed-form 
    # answer directly
    def apply_transform(self, in_transform):
        mean_new = np.zeros(in_transform.shape[0])
        mean_new = in_transform * self.mean
        self.mean = mean_new
        self.mean = self.mean[0:in_transform.shape[0]+1]

        between_var = np.zeros([in_transform.shape[1], in_transform.shape[1]])
        within_var = np.zeros([in_transform.shape[1], in_transform.shape[1]])
        psi_mat = np.zeros([in_transform.shape[1], in_transform.shape[1]])
        between_var_new = np.zeros([in_transform.shape[1], in_transform.shape[1]])
        within_var_new = np.zeros([in_transform.shape[1], in_transform.shape[1]])
        
        transform_invert = np.invert(self.transform)
        psi_mat = np.add(psi_mat, np.diag(self.psi))
        within_var = 1.0 * transform_invert * transform_invert.T
        between_var = 1.0 * transform_invert * psi_mat * transform_invert.T
        between_var_new = 1.0 * in_transform * between_var * in_transform.T
        within_var_new = 1.0 * in_transform * within_var * in_transform.T
        
        transform1 = compute_normalizing_transform(within_var_new)

        between_var_proj = 1.0 * transform1 * between_var_new * transform1.T

        s, U = scipy.linalg.eig(between_var_proj)
        s[s < 0] = 0
        U[U < 0] = 0
        s = np.sort(s)
        U = np.sort(U)

        self.transform = 1.0 * U * transform1 * U.T
        self.psi = s
        self.compute_derived_vars()

    def compute_derived_vars(self):
        self.offset = np.zeros(self.dim)
        self.offset = -1.0 * self.transform * self.mean
        return self.offset

    def get_normalization_factor(self, transform_ivector, num_example):
        transform_ivector_sq = transform_ivector
        transform_ivector_sq = np.power(transform_ivector,
                                        np.full(transform_ivector_sq.shape, 2.0))
        inv_covar = self.psi
        inv_covar = np.add(inv_covar,
                           np.full(inv_covar.shape, 1.0/num_example))
        inv_covar = np.reciprocal(inv_covar)
        dot_prob = inv_covar * transform_ivector_sq.T
        return dot_prob


class PldaEstimationConfig(object):
    def __init__(self, num_em_iters=10):
        self.num_em_iters = num_em_iters


class PldaEstimation(object):
    def __init__(self, Pldastats):
        self.stats = Pldastats
        self.dim = Pldastats.dim
        self.between_var = np.zeros(Pldastats.dim)
        self.between_var_stats = np.zeros(Pldastats.dim)
        self.between_var_count = 0
        self.within_var = np.zeros(Pldastats.dim)
        self.within_var_stats = np.zeros(Pldastats.dim)
        self.within_var_count = 0

    def estimate(self, config):
        for i in range(config.num_em_iters):
            print("Plda estimation %d of %d" % i, config.num_em_iters)
            self.estimate_one_iter()
    
    def compute_object_function_part1(self):
        within_class_count = self.stats.example_weight - self.stats.class_weight
        inv_within_var = self.within_var
        inv_within_var = np.invert(inv_within_var)
        _, within_logdet = np.linalg.slogdet(inv_within_var)
        objf = -0.5 * (within_class_count * (within_logdet + M_LOG_2PI * self.dim)
                        + np.trace(inv_within_var, self.stats.offset))
        return objf

    def compute_object_function_part2(self):
        tot_objf = 0.0
        n = -1
        for i in range(np.array(self.stats.classinfo).shape[0]):
            info = self.stats.classinfo[i]
            if info.num_example:
                combined_inv_var = self.between_var
                combined_inv_var += (1.0 / n) * self.within_var 
                _, combined_var_logdet = np.linalg.slogdet(np.invert(combined_inv_var))
                combined_inv_var = np.invert(combined_inv_var)
                mean = info.mean
                mean += -1.0/self.stats.class_weight * self.stats.sum
                tot_objf += info.weight * -0.5 * (combined_var_logdet + M_LOG_2PI * self.dim 
                                                  + mean.T * combined_inv_var * mean) 
                
    def estimate_one_iter(self):
        self.reset_per_iter_stats()
        self.get_stats_from_intraclass()
        self.get_stats_from_class_mean()
        self.estimate_from_stats()

    def compute_object_function(self):
        ans1 = self.compute_object_function_part1
        ans2 = self.compute_object_function_part2
        ans = ans1 + ans2
        example_weights = self.stats.example_weight
        #
        normalized_ans = ans / example_weights
        return normalized_ans

    def init_parameters(self):
        self.within_var = np.zeros(self.dim)
        self.between_var = np.zeros(self.dim)

    def reset_per_iter_stats(self):
        self.within_var_stats = np.zeros(self.stats.dim)
        self.within_var_count = 0
        self.between_var_stat = np.zeros(self.stats.dim)
        self.between_var_count = 0

    def get_stats_from_intraclass(self):
        self.within_var_stats += self.stats.offset_scatter
        self.within_var_count += self.stats.example_weight - self.stats.class_weight
    
    def get_stats_from_class_mean(self):
        between_var_inv = np.invert(self.between_var)
        within_var_inv = np.invert(self.within_var)
        mix_var = np.zeros(self.dim)
        for i in range(self.stats.num_classes):
            info = self.stats.classinfo[i]
            weight = info.weight
            if info.num_example:
                n = info.num_example
                mix_var = between_var_inv +  n * within_var_inv
                m = info.mean - (self.stats.sum / self.stats.class_weight)
                temp = n * within_var_inv * m
                w = mix_var * temp
                m_w = m - w
                self.between_var_stats += weight * mix_var
                self.between_var_stats += weight * np.square(w)
                self.between_var_count += weight
                self.within_var_stats += weight * n * mix_var
                self.within_var_stats += weight * n * np.square(m_w)
                self.within_var_count += weight

    def estimate_from_stats(self):
        self.within_var = (1.0 / self.within_var_count) * self.within_var_stats
        self.between_var = (1.0 / self.between_var_count) * self.between_var_stat

    def get_output(self, Plda_output):
        Plda_output.mean = (1.0 / self.stats.class_weight) * self.stats.mean
        transform1 = compute_normalizing_transform(self.within_var)
        between_var_proj = transform1 * self.between_var * transform1.T

        s, U = np.linalg.eigh(between_var_proj)
        sort_svd(s, U)
        Plda_output.transform = U.T
        Plda_output.psi = s
        #
        # tmp_within = Plda_output.transform * self.within_var * Plda_output.transform.T
        #TODO:Assert isunit
        # 
        # tmp_between = Plda_output.transform * self.between_var * Plda_output.transform.T

        Plda_output.compute_derived_vars()
        return Plda_output

class PldaUnsupervisedAdaptorConfig(object):
    def __init__(self, 
                 mean_diff_scale=1.0,
                 within_covar_scale=0.3,
                 between_covar_scale=0.7):
        self.mean_diff_scale = mean_diff_scale
        self.within_covar_scale = within_covar_scale
        self.between_covar_scale = between_covar_scale


class PldaUnsupervisedAdaptor(object):
    def __init__(self):
        self.tot_weight = 0
        self.mean_stats = np.zeros([])
        self.variance_stats = np.zeros([])
    
    def add_stats(self, weight, ivector):
        if self.mean_stats.shape[0] == 0:
            self.mean_stats = np.zeros(ivector.shape)
            self.variance_stats = np.zeros(ivector.shape)
        self.tot_weight += weight
        self.mean_stats += weight * ivector
        self.variance_stats += weight * np.square(ivector)
        
    def update_plda(self, config, plda):
        dim = self.mean_stats.shape[0]
        #TODO:Add assert
        mean = (1.0 / self.tot_weight) * self.mean_stats
        variance = (1.0 / self.tot_weight) * self.variance_stats - np.square(mean)
        plda.mean = mean
        transform_mod = plda.transform
        for i in range(dim):
            transform_mod[i] *= 1.0 / math.sqrt(1.0 + plda.psi[i])
        variance_proj = transform_mod * variance * transform_mod.T

        s, P = np.linalg.eigh(variance_proj)
        sort_svd(s, P)
        W = np.zeros([dim, dim])
        B = np.zeros([dim, dim])
        for i in range(dim):
            W[i][i] = 1.0 / (1.0 + plda.psi[i])
            B[i][i] = plda.psi[i] / (1.0 + plda.psi[i])
        Wproj2 = P.T * W * P
        Bproj2 = P.T * B * P
        Ptrans = P.T
        Wproj2mod = Wproj2
        Bproj2mod = Bproj2
        for i in range(dim):
            within = Wproj2[i][i]
            between = Bproj2[i][i]
            # Problem
            if s[i] > 1.0:
                excess_eig = s[i] - 1.0
                excess_within_covar = excess_eig * config.within_covar_scale
                excess_between_covar = excess_eig * config.between_covar_scale
                Wproj2mod[i][i] += excess_within_covar
                Bproj2mod[i][i] += excess_between_covar
        combined_trans_inv = np.invert(Ptrans * transform_mod)
        Wmod = combined_trans_inv * Wproj2mod * combined_trans_inv.T
        Bmod = combined_trans_inv * Bproj2mod * combined_trans_inv.T
        C_inv = np.invert(np.linalg.cholesky(Wmod))
        Bmod_proj = C_inv * Bmod * C_inv.T
        psi_new, Q = np.linalg.eigh(Bmod_proj)
        sort_svd(psi_new, Q)
        final_transform = Q.T * C_inv
        plda.transform = final_transform
        plda.psi = psi_new


def compute_normalizing_transform(covar):
    c = np.linalg.cholesky(covar)
    c = np.invert(c)
    return c

def sort_svd(s, d):
    for i in len(s)-1:
        j = i
        while j<len(s)-1:
            if s[j] < s[j+1]:
                tmp = s[j]
                s[j] = s[j+1]
                s[j+1] = tmp
                tmp = d[j]
                d[j] = d[j+1]
                d[j+1] = tmp
    return s, d
    
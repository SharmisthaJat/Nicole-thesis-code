import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import load_data
import agg_TGM


if __name__ == '__main__':
    exp = 'krns2'
    mode = 'pred'
    word = 'firstNoun'
    sen_type = 'active'
    accuracy = 'abs'
    sub_results = {}
    sub_params = {}
    sub_time = {}
    for sub in ['B', 'D']:  # load_data.VALID_SUBS[exp]:
        param_specs = {'o': 12,
                       'pd': 'F',
                       'pr': 'F',
                       'alg': 'LR',
                       'F': 2,
                       'z': 'F',
                       'avg': 'F',
                       'ni': 2,
                       'nr': 10,
                       'rs': 1}
        sub_results[sub], sub_params[sub], sub_time[sub] = agg_TGM.agg_results(exp,
                                                                               mode,
                                                                               word,
                                                                               sen_type,
                                                                               accuracy,
                                                                               sub,
                                                                               param_specs=param_specs)
    diag, param_val, time = agg_TGM.get_diag_by_param(sub_results, sub_params, sub_time, 'w', {})

    diag = np.mean(diag, axis=0)

#    fig, ax = plt.subplots()
  #  im = ax.imshow(diag, interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    #plt.colorbar(im)
    plt.plot(diag[0, :])
    plt.show()




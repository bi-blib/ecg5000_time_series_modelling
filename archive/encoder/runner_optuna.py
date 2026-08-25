try:
    from ...src.encoder import abnormal_multiclassifer_optuna as amc_optuna
    from ...src.encoder import normal_abnormal_binary_classifier_optuna as nabc_optuna
except ImportError:
    import ecg5000_time_series_modelling.archive.encoder.abnormal_multiclassifer_optuna as amc_optuna
    import ecg5000_time_series_modelling.archive.encoder.normal_abnormal_binary_classifier_optuna as nabc_optuna

nabc_optuna.main()
amc_optuna.main()

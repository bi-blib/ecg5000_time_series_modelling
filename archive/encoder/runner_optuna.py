try:
    import abnormal_multiclassifer_optuna as amc_optuna
    import normal_abnormal_binary_classifier_optuna as nabc_optuna
except ImportError:
    import abnormal_multiclassifer_optuna as amc_optuna
    import normal_abnormal_binary_classifier_optuna as nabc_optuna

nabc_optuna.main()
amc_optuna.main()

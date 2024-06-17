def get_default_config(data_name):
    if data_name == 'dro':
        return dict(
            dims=[83, 256, 128],
            pretrain_epochs=180,
            epochs=850,
            pre_lr=0.001,
            lr=0.001,
            batch_size=512,
            weight_map=1,
            weight_coef=1,
            weight_ent=0,
            alpha=1,
            ot=dict(
                epochs=500,
                lr=0.05,
                step_size=5000,
                tau=1,
                it=3,
                epsilon=1,
                num_iter=5,
                g1=1,
                g2=1
            )
        )
    elif data_name == 'BRCA':
        return dict(
            dims=[19736, 4096, 2048],
            pretrain_epochs=100,
            epochs=123,
            pre_lr=0.0001,
            lr=0.0005,
            batch_size=128,
            weight_map=1,
            weight_coef=1,
            weight_mmd=0,
            weight_ent=0,
            alpha=1,
            ot=dict(
                epochs=300,
                lr=0.05,
                step_size=300,
                tau=1,
                it=3,
                epsilon=1,
                num_iter=5,
            )

        )
    elif data_name == 'MERFISH':
        return dict(
            dims=[232, 512, 256],
            pretrain_epochs=80,
            epochs=20,
            pre_lr=0.0005,
            lr=0.0005,
            batch_size=256,
            weight_map=10,
            weight_coef=0,
            weight_mmd=0,
            weight_ent=0,
            alpha=1,
            ot=dict(
                epochs=300,
                lr=0.05,
                step_size=300,
                tau=1,
                it=3,
                epsilon=1,
                num_iter=5,
            )
        )
    elif data_name == 'STARmap':
        return dict(
            dims=[2222, 512, 256],
            pretrain_epochs=400,
            epochs=15,
            pre_lr=0.0001,
            lr=0.0001,
            batch_size=256,
            weight_map=10,
            weight_coef=0,
            weight_mmd=0,
            weight_ent=0,
            alpha=1,
        )

    else:
        raise Exception('Undefined data_name')

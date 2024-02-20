cfg =dict(
    seed =2022,
     train_dir =r"/data/pacm_dataset/train", 
    train_label_dir =r"/data/pacm_dataset/train.csv",

    valid_dir  =r"/data/pacm_dataset/valid",
    valid_label_dir =r"/data/pacm_dataset/valid.csv",

    test_dir =r"/data/pacm_dataset/test",
    test_label_dir =r"/data/pacm_dataset/test.csv",


    model_path="/parameter.pkl",
    image_size =224,
    num_epochs =60,
    batch_size =32,
    display_step =1,
    adversarial_criterion =nn.BCEWithLogitsLoss(),
    recon_criterion =nn.L1Loss(),
    lambda_recon =100,
)
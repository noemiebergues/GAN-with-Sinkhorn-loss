# Set up the parameters
seed = 1126
image_size = 128
batch_size = 4
nz = 128
nc = 3
lr =0.000005
max_iter = 10
epsilon = 10
niter_sink = 1000
truncation = 0.2
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


"""
Van gogh Dataset
"""
#Upload the data
data_dir='/content/drive/MyDrive/paysages'
train_ds = ImageFolder(data_dir, transform=T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize(*stats)]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)


# construct GAN
#G_decoder = G
D_encoder = Encoder(image_size, nc, k=128, ndf=128)
D_decoder = Decoder(image_size, nc, k=128, ngf=128)


netG=BigGAN.from_pretrained('biggan-deep-128')
netD = NetD(D_encoder, D_decoder)
netD.apply(weights_init)


# put variable into cuda device
netG.cuda()
netD.cuda()


# setup optimizer
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=0.00001)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=0.000005)

# Train
path_save = '/content/drive/MyDrive/gan'
train_gan(train_dl, 1000, truncation,epsilon, niter_sink,path_save)

"""
MNIST Dataset
"""
#Upload the data
def get_data(dataroot, image_size, train_flag=True):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            ( 0.5), ( 0.5)),
    ])

    dataset = MNIST(root= dataroot,
                             download=True,
                             train=train_flag,
                             transform=transform
                    )

    return dataset
dataroot = os.getcwd()
trn_dataset = get_data(dataroot, image_size, train_flag=True)
trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)

#parameters
seed = 1126
image_size = 64
batch_size = 200
nz = 100
nc = 1
lr =0.0005
max_iter = 10
epsilon = 1
niter_sink = 100
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# construct GAN
G_decoder = Decoder(image_size, nc, k= nz, ngf=64)
netG = NetG(G_decoder)
netG.apply(weights_init)
D_encoder = Encoder(image_size, nc, k=100, ndf=100)
D_decoder = Decoder(image_size, nc, k=100, ngf=100)
netD = NetD(D_encoder, D_decoder)
netD.apply(weights_init)

# put variable into cuda device
netG.cuda()
netD.cuda()

# setup optimizer
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=lr)

#Train
train_gan_mnist(trn_loader, 100, epsilon, niter_sink, 50)


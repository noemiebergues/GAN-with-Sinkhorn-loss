"""
Function to train the GAN
"""

def train_gan(train_dl, n_epoch, truncation,epsilon, niter_sink, path_save):
  gen_iterations = 0
  D_loss = []
  G_loss = []
  noise_vector2 = truncated_noise_sample(truncation=truncation, batch_size=4)
  noise_vector2 = torch.from_numpy(noise_vector2)
  noise_vector2 = noise_vector2.to('cuda')

  class_vector2 = one_hot_from_int(0, batch_size=4)
  class_vector2 = torch.from_numpy(class_vector2)
  class_vector2 = class_vector2.to('cuda')

  for t in range(n_epoch):
    print('epoch_',t+300)
    y=netG(noise_vector2,class_vector2 , truncation = truncation)
    show_image(y)
    data_iter = iter(train_dl)
    torch.save(netG,path_save+'/netG_')
    i = 0
    while (i < len(train_dl)):
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = True
        
        for p in netG.parameters():
          p.requires_grad = False

        # clamp parameters of NetD encoder to a cube
        # do not clamp paramters of NetD decoder!!!
        for p in netD.encoder.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        data = next(data_iter)
        i += 1
        netD.zero_grad()

        x_cpu, label = data
        x = x_cpu.cuda()
        batch_size = x.size(0)

        f_enc_X_D, f_dec_X_D = netD(x)

        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
        noise_vector = torch.from_numpy(noise_vector)
        noise_vector = noise_vector.to('cuda')

        class_vector = one_hot_from_int(label, batch_size=batch_size)
        class_vector = torch.from_numpy(class_vector)
        class_vector = class_vector.to('cuda')

        y = netG(noise_vector,class_vector , truncation = truncation)
        f_enc_Y_D, f_dec_Y_D = netD(y)
      

        sink_D = 2*my_sinkhorn(f_enc_X_D, f_enc_Y_D, epsilon,batch_size,niter_sink) \
                - my_sinkhorn(f_enc_Y_D, f_enc_Y_D, epsilon, batch_size,niter_sink) \
                -my_sinkhorn(f_enc_X_D, f_enc_X_D, epsilon, batch_size,niter_sink)
        errD = -sink_D

        errD.backward()
        optimizerD.step()

        # ---------------------------
        #        Optimize over NetG
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = False
          
        for p in netG.parameters():
          p.requires_grad = True

        netG.zero_grad()
        if (i == len(train_dl)):
          break
        data = next(data_iter)
        i += 1
        x_cpu, label = data
        x = x_cpu.cuda()
        batch_size = x.size(0)

        f_enc_X, f_dec_X = netD(x)

        noise_vector = truncated_noise_sample(truncation=truncation, batch_size=batch_size)
        noise_vector = torch.from_numpy(noise_vector)
        noise_vector = noise_vector.to('cuda')

        class_vector = one_hot_from_int(label, batch_size=batch_size)
        class_vector = torch.from_numpy(class_vector)
        class_vector = class_vector.to('cuda')

        y=netG(noise_vector,class_vector , truncation = truncation)
        f_enc_Y, f_dec_Y = netD(y)

        ###### Sinkhorn loss #########
        sink_G = 2*my_sinkhorn(f_enc_X, f_enc_Y, epsilon,batch_size,niter_sink) \
                - my_sinkhorn(f_enc_Y, f_enc_Y, epsilon, batch_size,niter_sink) \
                - my_sinkhorn(f_enc_X, f_enc_X, epsilon, batch_size,niter_sink)
        errG = sink_G

        errG.backward()
        optimizerG.step()
        gen_iterations += 1

        D_loss.append(sink_D)
        G_loss.append(sink_G)
    return D_loss, G_loss

def train_gan_mnist(train_dl, n_epoch, epsilon, niter_sink, affichage):
  gen_iterations = 0
  D_loss = []
  G_loss = []

  for t in range(n_epoch):
    print('epoch_',t)
    data_iter = iter(train_dl)
    i = 0
    while (i < len(train_dl)):
        # ---------------------------
        #        Optimize over NetD
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = True
        
        for p in netG.parameters():
          p.requires_grad = False

        # clamp parameters of NetD encoder to a cube
        # do not clamp paramters of NetD decoder!!!
        for p in netD.encoder.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        data = next(data_iter)
        i += 1
        netD.zero_grad()

        x_cpu, label = data
        x = x_cpu.cuda()
        batch_size = x.size(0)

        f_enc_X_D, f_dec_X_D = netD(x)

        noise_vector = torch.cuda.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)

        y = netG(noise_vector)
        f_enc_Y_D, f_dec_Y_D = netD(y)
      

        sink_D = 2*my_sinkhorn(f_enc_X_D, f_enc_Y_D, epsilon,batch_size,niter_sink) \
                - my_sinkhorn(f_enc_Y_D, f_enc_Y_D, epsilon, batch_size,niter_sink) \
                -my_sinkhorn(f_enc_X_D, f_enc_X_D, epsilon, batch_size,niter_sink)
        errD = - sink_D

        errD.backward()
        optimizerD.step()

        # ---------------------------
        #        Optimize over NetG
        # ---------------------------
        for p in netD.parameters():
            p.requires_grad = False
          
        for p in netG.parameters():
          p.requires_grad = True

        netG.zero_grad()

        #data = next(data_iter)
        #i += 1
        #x_cpu, label = data
        #x = x_cpu.cuda()
        #batch_size = x.size(0)

        f_enc_X, f_dec_X = netD(x)

        noise_vector = torch.cuda.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)

        y=netG(noise_vector)
        f_enc_Y, f_dec_Y = netD(y)

        ###### Sinkhorn loss #########
        sink_G = 2*my_sinkhorn(f_enc_X, f_enc_Y, epsilon,batch_size,niter_sink) \
                - my_sinkhorn(f_enc_Y, f_enc_Y, epsilon, batch_size,niter_sink) \
                - my_sinkhorn(f_enc_X, f_enc_X, epsilon, batch_size,niter_sink)
        errG = sink_G

        errG.backward()
        optimizerG.step()
        gen_iterations += 1

        D_loss.append(-sink_D)
        G_loss.append(sink_G)

        if i% affichage==0 :
          print('issue_du_batch',i,'errD=',errD)
          print('issue_du_batch',i,'errG=',errG)
          show_tensor_images(y)


    return D_loss, G_loss

import time
from ops import *
import numpy as np
from utils import *
from glob import glob
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch

class FBDE(object):

    def __init__(self, sess, args):
    
        self.sess = sess
        self.phase = args.phase
        self.model_name = 'FBDE'
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.result_dir = args.result_dir
        self.augment_flag = args.augment_flag
        self.checkpoint_dir = args.checkpoint_dir

        self.epoch = args.epoch
        self.gan_type = args.gan_type
        self.save_freq = args.save_freq
        self.iteration = args.iteration

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq

        self.ch = args.ch
        self.nce_temp = args.nce_temp
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch
        self.init_lr_for_g = args.lr_for_g
        self.init_lr_for_d = args.lr_for_d
        
        """ Weight """
        self.margin = args.margin
        self.smoothing = args.smoothing
        self.adv_weight = args.adv_weight
        self.num_patches = args.num_patches
        self.pixel_weight = args.pixel_weight
        self.contrastive_weight = args.contrastive_weight

        """ net_G """
        self.n_res = args.n_res

        """ net_D """
        self.sn = args.sn
        self.n_dis = args.n_dis
        self.img_ch = args.img_ch
        self.n_critic = args.n_critic
        self.img_size = args.img_size

        """ Sample dir. """
        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        
        """ Dataset """
        self.trainA_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainA'))
        self.trainB_dataset = glob('./dataset/{}/*.*'.format(self.dataset_name + '/trainB'))
        self.dataset_num = min(len(self.trainA_dataset), len(self.trainB_dataset))

        print()

        print("##### Information #####")
        print("# epoch : ", self.epoch)
        print("# gan type : ", self.gan_type)
        print("# smoothing : ", self.smoothing)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# min dataset number : ", self.dataset_num)

        print()

        print("##### net_G #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### net_D #####")
        print("# discriminator layer : ", self.n_dis)
        print("# spectral normalization : ", self.sn)
        print("# the number of critic : ", self.n_critic)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)

    ##################################################################################
    # GENERATOR.
    ##################################################################################
            
    def net_G(self, x_init, reuse=False, scope="generator"):
        channel = self.ch
        
        with tf.variable_scope(scope, reuse=reuse):
            """ ENCODER """
            # (256,256,3) -> (256,256,64)
            x = conv(x_init, channel, kernel = 7, stride = 1, pad = 3, pad_type ='reflect', scope='conv_0')
            x = instance_norm(x, scope='ins_norm_0')
            x = relu(x)
            
            # Down-Sampling. (256,256,64) -> (128,128,128) -> (64,64,256)
            for i in range(1, 3) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i))
                x = instance_norm(x, scope='ins_norm_'+str(i))
                x = relu(x)
                channel = channel * 2
            
            # Down-Sampling Bottleneck. (64,64,256) -> (64,64,256)
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_'+str(i))
            
            # Class Activation Map. (64,64,256) -> (64,64,512) -> (64,64,256)
            cam_x = global_avg_pooling(x)
            _, cam_gap_weight = fully_connected_with_w(cam_x, reuse=False, scope='cam_logit')
            cam_gap = tf.multiply(x, cam_gap_weight)
            cam_x = global_max_pooling(x)
            _, cam_gmp_weight = fully_connected_with_w(cam_x, reuse=True, scope ='cam_logit')
            cam_gmp = tf.multiply(x, cam_gmp_weight)
            cam_map = tf.concat([cam_gap, cam_gmp], axis=-1)
            
            x = conv(cam_map, channel, kernel=1, stride=1, scope='conv_1x1')
            x = relu(x)

            # Gamma, Beta block.
            style = global_avg_pooling(x)
            gamma, beta = self.MLP(style, reuse=reuse)
            
            # Up-Sampling Bottleneck.
            for i in range(self.n_res):
                x = adaptive_ins_layer_resblock(x, channel, gamma, beta, 
                                                    smoothing=self.smoothing, scope='adaptive_resblock'+str(i))

            # Up-Sampling.
            for i in range(2):
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='up_conv_'+str(i))
                x = layer_instance_norm(x, scope='layer_ins_norm_'+str(i))
                x = relu(x)
                
                channel = channel // 2
                
            x = conv(x, channels = 3, kernel = 7, stride = 1, pad=3, pad_type='reflect', scope='g_logit')
            x = tanh(x)
            
            return x

    def MLP(self, x, use_bias=True, reuse=False, scope='MLP'):
    
        channel = self.ch * self.n_res
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2):
                x = fully_connected(x, channel, use_bias, scope='linear_'+str(i))
                x = relu(x)
            
            gamma = fully_connected(x, channel, use_bias, scope = 'gamma')
            beta  = fully_connected(x, channel, use_bias, scope =  'beta')
            
            gamma = tf.reshape(gamma, shape=[self.batch_size, 1, 1, channel])
            beta  = tf.reshape(beta, shape =[self.batch_size, 1, 1, channel])
                    
            return gamma, beta

    ##################################################################################
    # Embeeding NET.
    ##################################################################################
                 
    def net_F(self, _input, ids=None, reuse=False, scope='mlp'):
        
        channel = self.ch * self.n_res
        with tf.variable_scope(scope, reuse=reuse):
        
            B, H, W, C = _input.shape            
            feat_reshape = tf.reshape(_input, [B, -1, C])

            if ids is not None:
                id_s = ids
            else:
                id_s = tf.random_shuffle(tf.range(H * W))[:min(self.num_patches, H * W)]
                
            # Per-patch fully-connected layer.
            _input = tf.squeeze(tf.gather(feat_reshape, id_s, axis=1), axis=0)
            _input = fully_connected(_input, channel, scope='input_embbeding')
            _input = relu(_input)
            
            # multi mlp layers.
            _input = fully_connected(_input, channel, scope='chann_embbeding')
            _input = relu(_input)
            
            # output fully-connected layer.
            _input = fully_connected(_input, channel, scope='outpu_embbeding')
            _input = tf.nn.l2_normalize(_input, axis=-1)

        return _input, id_s

    ##################################################################################
    # MULTI SCALES DISCRIMINATOR.
    ##################################################################################

    def multi_net_D(self, x_init, reuse=False, scope='discriminator'):
        D_logit = []
        D_CAM_logit = []
        
        with tf.variable_scope(scope, reuse=reuse) :
            for scale in range(3):
                channel = self.ch
                x = conv(x_init, channel, kernel = 4, stride = 2, pad = 1, pad_type='reflect', sn=self.sn, scope ='ms_'+str(scale)+'_conv_0')
                x = lrelu(x, 0.2)
                
                for i in range(1, self.n_dis - 1):
                    x = conv(x, channel*2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='ms_'+str(scale)+'_conv_'+str(i))
                    x = lrelu(x, 0.2)
                    channel = channel * 2
                    
                x = conv(x, channel * 2, kernel = 4, stride = 1, pad=1, pad_type='reflect', sn =self.sn, scope='ms_'+str(scale)+'_conv_last')
                x = lrelu(x, 0.2)
                channel = channel * 2
            
                # Class Activation Map.
                cam_x = global_avg_pooling(x)
                cam_gap_logit, cam_gap_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=False, scope='ms_'+str(scale)+'_cam_logit')
                cam_gap = tf.multiply(x, cam_gap_weight)
                cam_x = global_max_pooling(x)
                cam_gmp_logit, cam_gmp_weight = fully_connected_with_w(cam_x, sn=self.sn, reuse=True, scope ='ms_'+str(scale)+'_cam_logit')
                cam_gmp = tf.multiply(x, cam_gmp_weight)
                cam_map = tf.concat([cam_gap, cam_gmp], axis=-1)
                
                x = conv(cam_map, channel, kernel=1, stride=1, scope='ms_'+str(scale)+'_conv_1x1')
                x = lrelu(x, 0.2)
                
                cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis = -1)
                pat_logit = conv(x, channels=1, kernel=4, stride=1, pad=1, pad_type='reflect', sn=self.sn, scope='ms_'+str(scale)+'_d_logit')
                
                D_logit.append(pat_logit)
                D_CAM_logit.append(cam_logit)
                
                x_init = down_sample(x_init)

            return D_logit, D_CAM_logit

    ##################################################################################
    # BUILD FBDE MODEL.
    ##################################################################################
    
    def build_model(self):
    
        if self.phase == 'train':

            """ Load images for G, D """
            Image_Data_Class = ImageData(self.img_size, self.img_ch, self.augment_flag)
            trainA = tf.data.Dataset.from_tensor_slices(self.trainA_dataset)
            trainB = tf.data.Dataset.from_tensor_slices(self.trainB_dataset)
            
            gpu_device = '/gpu:1'
            trainA = trainA.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, 
                                  self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))
            trainB = trainB.apply(shuffle_and_repeat(self.dataset_num)).apply(map_and_batch(Image_Data_Class.image_processing, 
                                  self.batch_size, num_parallel_batches=16, drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

            self.real_A = trainA.make_one_shot_iterator().get_next()
            self.real_B = trainB.make_one_shot_iterator().get_next()
            
            """ Load parameters for net_G, net_D and net_F """
            self.lr_for_g = tf.placeholder(tf.float32, name='learning_rate_for_g')
            self.lr_for_d = tf.placeholder(tf.float32, name='learning_rate_for_d')

            """ Define Generator, Discriminator and net_F """
            # Fake b content logits.
            self.fake_B = self.net_G(self.real_A, reuse=False, scope="generator")
            
            # Real A and Fake B mlp logits.
            feat_A, ids = self.net_F(self.real_A, ids=None, reuse=False, scope='net_F')
            feat_B, ids = self.net_F(self.fake_B, ids=ids, reuse=True, scope='net_F')
            
            # Real B and Fake B discriminator logits.
            realB, realB_cam = self.multi_net_D(self.real_B, reuse=False, scope="discriminator")
            fakeB, fakeB_cam = self.multi_net_D(self.fake_B, reuse=True,  scope="discriminator")
            
            ''' Define loss for net_G, net_D and net_F. '''
            # net_G sty loss.
            net_G_gan_loss = self.pixel_weight * histogram_loss(self.fake_B, self.real_B)
            
            # net_G adv loss.
            net_G_adv_loss =  self.adv_weight * (gloss(self.gan_type, fakeB) + gloss(self.gan_type, fakeB_cam))
            
            # net_F nce Loss.
            self.net_F_loss = self.contrastive_weight * contrastive_loss(feat_A, feat_B, temperature=self.nce_temp)
            
            # net_D adv loss.
            net_D_adv_loss =  self.adv_weight * (dloss(self.gan_type, realB, fakeB) + dloss(self.gan_type, realB_cam, fakeB_cam))
            
            # Total adv loss.
            self.net_G_loss = net_G_adv_loss + net_G_gan_loss + self.net_F_loss + reg_loss('generator')
            self.net_D_loss = net_D_adv_loss + reg_loss('discriminator')
            
            """ Training model """
            t_vars = tf.trainable_variables()
            self.net_F_vars = [var for var in t_vars if 'net_F' in var.name]
            self.net_G_vars = [var for var in t_vars if 'generator' in var.name]
            self.net_D_vars = [var for var in t_vars if 'discriminator' in var.name]
            
            self.net_G_optim = tf.train.AdamOptimizer(self.lr_for_g, beta1=0.5, beta2=0.999).minimize(self.net_G_loss, var_list=self.net_G_vars)
            self.net_F_optim = tf.train.AdamOptimizer(self.lr_for_d, beta1=0.5, beta2=0.999).minimize(self.net_F_loss, var_list=self.net_F_vars)
            self.net_D_optim = tf.train.AdamOptimizer(self.lr_for_d, beta1=0.5, beta2=0.999).minimize(self.net_D_loss, var_list=self.net_D_vars)
            
            """" Summary scalar """
            self.all_net_G_loss = tf.summary.scalar("total_net_G_loss", self.net_G_loss)
            self.all_net_F_loss = tf.summary.scalar("total_net_F_loss", self.net_F_loss)
            self.all_net_D_loss = tf.summary.scalar("total_net_D_loss", self.net_D_loss)
            
            # Summary List.
            self.rho_var = []
            for var in tf.trainable_variables():
                if 'rho' in var.name:
                    self.rho_var.append(tf.summary.histogram(var.name, var))
                    self.rho_var.append(tf.summary.scalar(var.name + "_min",  tf.reduce_min(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_max",  tf.reduce_max(var)))
                    self.rho_var.append(tf.summary.scalar(var.name + "_mean",tf.reduce_mean(var)))

            g_summary_list = [self.all_net_G_loss]
            f_summary_list = [self.all_net_F_loss]
            d_summary_list = [self.all_net_D_loss]
            g_summary_list.extend(self.rho_var)

            self.net_G_summary = tf.summary.merge(g_summary_list)
            self.net_F_summary = tf.summary.merge(f_summary_list)
            self.net_D_summary = tf.summary.merge(d_summary_list)

        else :
            """ Test model """
            self.test_domain_A = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
            self.test_fake_B = self.net_G(self.test_domain_A, reuse=False, scope="generator")

    def train(self):
        # initialize all variables.
        tf.global_variables_initializer().run()

        # saver to save model.
        self.saver = tf.train.Saver()

        # summary writer.
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)
        
        # restore check-point if it exits.
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch.
        start_time = time.time()
        past_g_loss = -1.
                    
        for epoch in range(start_epoch, self.epoch):
            
            if self.decay_flag:					
                lr_for_g = self.init_lr_for_g if epoch < self.decay_epoch else (
                                                        self.init_lr_for_g * (self.epoch - epoch) / (self.epoch - self.decay_epoch))
                lr_for_d = self.init_lr_for_d if epoch < self.decay_epoch else (
                                                        self.init_lr_for_d * (self.epoch - epoch) / (self.epoch - self.decay_epoch))
                                                        
            batch_idxs = self.iteration // self.batch_size
            for idx in range(start_batch_id, batch_idxs):
                train_feed_dict_for_g = {self.lr_for_g : lr_for_g}
                train_feed_dict_for_d = {self.lr_for_d : lr_for_d}
                
                # Update net_G.
                g_loss = None
                if (counter - 1) % self.n_critic == 0:
                    _, g_loss, fake_B, real_A, summary_str = self.sess.run([self.net_G_optim, self.net_G_loss, self.fake_B, self.real_A, 
                                                                                           self.net_G_summary], feed_dict = train_feed_dict_for_g)
                    self.writer.add_summary(summary_str, counter)
                    past_g_loss = g_loss

                # Update net_F.
                _, c_loss, summary_str = self.sess.run([self.net_F_optim, self.net_F_loss, self.net_F_summary], feed_dict = train_feed_dict_for_d)
                self.writer.add_summary(summary_str, counter)

                # Update net_D.
                _, d_loss, summary_str = self.sess.run([self.net_D_optim, self.net_D_loss, self.net_D_summary], feed_dict = train_feed_dict_for_d)
                self.writer.add_summary(summary_str, counter)

                # display training status.
                counter += 1
                if g_loss == None:
                    g_loss = past_g_loss
                print("Epoch: [%2d] [%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f, c_loss: %.8f"% (epoch, idx, self.iteration, 
                                                                                  time.time() - start_time, d_loss, g_loss, c_loss))
                
                # sample images to visualize.
                if np.mod(idx+1, self.print_freq) == 0:
                    save_images(real_A, [self.batch_size, 1], './{}/real_A_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))
                    save_images(fake_B, [self.batch_size, 1], './{}/fake_B_{:03d}_{:05d}.png'.format(self.sample_dir, epoch, idx+1))

                # save model for every save freq.
                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)
                    
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model.
            start_batch_id = 0

            # save model for final step.
            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'

        if self.smoothing :
            smoothing = '_smoothing'
        else :
            smoothing = ''

        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        return "{}_{}_{}_{}_{}_{}_{}_{}{}".format(self.model_name, self.dataset_name, self.gan_type, 
                                                  n_res, n_dis, self.n_critic, self.adv_weight, sn, smoothing)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter   = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison.
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_domain_A : sample_image})
            save_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (
            sample_file if os.path.isabs(sample_file) else ('../..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (
               image_path if os.path.isabs(image_path) else ('../..' + os.path.sep + image_path), self.img_size, self.img_size))
            
            index.write("</tr>")
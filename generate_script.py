def generate_sbatch_ecog_encoder_singularity_lossweight_LD_multi(MAPPING_FROM_ECOG,OUTPUT_DIR,\
                ld_loss_weight, alpha_loss_weight,consonant_loss_weight,formant_supervision = 1,\
                    noisedb=-50,n_filter_samples=80, n_fft=256,reverse_order=1,pitch_supervision=0,\
                             lar_cap=0,intensity_thres = -1,bgnoisefromdata=1,filter_dirty_sample=0,\
                subject='NY742',RNN_LAYER=0,batch_size=2,task='active',learnedmask=1, dynamicfiltershape=0,\
                mapping_layers=0,region_index = 0,multiscale=0,trainsubject='', testsubject='',rdropout=0,\
                epoch_num=60,pretrained_model_dir = '', causal=0,job_abbrev='mul',use_stoi=0,FAKE_LD=0,extra_up=0,\
                use_denoise = 0,extend_grid=0,occlusion=0,hour=37,minute=59,hidden_dim = 256,s_dim=128,gradient_clip=0,\
                trilinear = 0, activation = 'leaky_relu',losses='gaussian',gpu='',downtimes=4, dim_selection =0,density='LD',\
                ctc_decoder='greedy',remove_target_duplicates=0,n_outputs=101,align_loss=0,infer = 0,n_features=64,\
                        regress_hubert=0,use_mixed_signal='None',pad_audio=0,loss_name='CE',brain='all',\
                                                           temporal_masking=0,distance_weighted=0):
    
    text = '#!/bin/bash'
    text += '\n'
    text += '\n'
    text += '#SBATCH --job-name={}{}'.format(job_abbrev,trainsubject[2:])
    text += '\n'
    text += '#SBATCH --nodes=1'
    text += '\n'
    text += '#SBATCH --cpus-per-task=1'
    text += '\n'
    text += '#SBATCH --mem=36GB'
    text += '\n'
    text += '#SBATCH --time={}:{}:00'.format(hour,minute)
    text += '\n'
    text += '#SBATCH --gres=gpu:{}1'.format(gpu)
    text += '\n'
    text += 'overlay_ext3=/scratch/xc1490/apps/overlay-50G-10M.ext3'
    text +='\n'
    text +='singularity exec --nv \\'
    text +='\n'
    text +='    --overlay ${overlay_ext3}:ro \\'
    text +='\n'
    text +='    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \\'
    text +='\n'
    text +='    /bin/bash -c "'
    text +='\n'
    text +='source /ext3/env.sh'
    text +='\n'
    text +='cd /scratch/xc1490/projects/ecog/hubert_decoding'
    text +='\n'
    #'occlusion_1122/ResNet/ResNet'
    trainnum = ','.join([str(i) for i in np.where(np.isin(depth_samples,trainsubject.split(','))==1)[0]])
    testnum = 'all'#','.join([str(i) for i in np.where(np.isin(depth_samples,testsubject.split(','))==1)[0]])
    output_dir = '/scratch/xc1490/projects/ecog/hubert_decoding/output/{}_e_{}_cau_{}_tr_{}_tst_{}_ta_{}_bs_{}_filter_{}_gradcl_{}_ext_up_{}_rmd_{}_nout_{}_align_{}_infer_{}_reg_{}_nfeat_{}_mixsig_{}_padaud_{}_losnam_{}_brain_{}_tmask_{}_dweigh_{}'.format(\
                    OUTPUT_DIR, epoch_num,causal,trainnum, \
                    testnum,   task, batch_size, filter_dirty_sample,gradient_clip,\
                    extra_up,remove_target_duplicates,n_outputs ,align_loss,infer,regress_hubert,n_features,\
                    use_mixed_signal,pad_audio,loss_name,brain,temporal_masking,distance_weighted)
    os.makedirs(output_dir,exist_ok = True)
    #print (output_dir)
    #print ('no files',len([i for i in os.listdir(output_dir) if i.endswith('.mat')] ))
    
    if len([i for i in os.listdir(output_dir) if i.endswith('.npy')] )<10:
        text += 'python train_e2a_multi.py --OUTPUT_DIR {} --trainsubject {} --testsubject {} --param_file {} --batch_size {} \
        --MAPPING_FROM_ECOG {} --DENSITY {} --pretrained_model_dir {} --n_fft {}  --epoch_num {} --ctc_decoder {} \
        --filter_dirty_sample {} --gradient_clip {} --extra_up {} --remove_target_duplicates {} --n_outputs {} \
        --align_loss {} --infer {} --regress_hubert {} --n_features {} --use_mixed_signal {} --pad_audio {} \
        --loss_name {} --brain {} --temporal_masking {} --causal {} --distance_weighted {}"'.format(\
                        output_dir, trainsubject, testsubject,   \
                        'configs/train_param_production.json'.format(task), batch_size, MAPPING_FROM_ECOG,density,\
                         pretrained_model_dir,n_fft ,epoch_num,ctc_decoder,filter_dirty_sample,gradient_clip,extra_up,\
                         remove_target_duplicates,n_outputs ,align_loss,infer,regress_hubert,n_features,use_mixed_signal,\
                         pad_audio,loss_name,brain,temporal_masking,causal,distance_weighted)
        text +='\n'

        job_file =  '/scratch/xc1490/projects/ecog/hubert_decoding/jobs/hubt_{}_tr_{}_tst_{}_tsk_{}_bs_{}_{}_ctc_{}_filt_{}_grdcl_{}_extp_{}_rmdp_{}_nout_{}_algn_{}_inf_{}_reg_{}_n_fet_{}_mxsgnl_{}_pad_{}_los_{}_brn_{}_tmask_{}_dwegh_{}.sbatch'.format(causal ,trainnum,\
                    testnum, task,batch_size, \
                    OUTPUT_DIR.split('/')[-1], ctc_decoder,filter_dirty_sample ,gradient_clip,extra_up,\
                    remove_target_duplicates,n_outputs ,align_loss,infer,regress_hubert,n_features,\
                    use_mixed_signal,pad_audio,loss_name,brain,temporal_masking,distance_weighted)
        f= open(job_file,"w+")
        f.write(text)
        f.close()
        print ('sbatch ',job_file)
        return '\n'.join(text.split('\n')[-3:])[:-2] +'\n'
       
    else:
        print ('finished!')
        return '\n'
# save train as well
#ctc loss zero infinity
samples_all = depth_samples
ld_loss_weight,alpha_loss_weight,consonant_loss_weight = 1,1,0
mapping_layer = 0
region_index = 0
multiscale = 0
hidden_dim = 128
count = 0
epoch_num = 4000

#allsub = ['NY758','NY763','NY765','NY798','NY829','NY749', \
#                    'NY748','NY836']
n_outputs = 100 
align_loss = 1
pad_audio = 0


for extra_up in [0 ,1  ]: #'ECoGMapping_SWIN_maxpool_aligned',
            for ecogmappingname in ['classifier_GRU_w_conv_aligned']:# 'classifier_GRU_w_conv' ]:#,]: 
                for filter_dirty_sample in [0 ]:
                    for use_mixed_signal  in [ 'None' ]: #,
                        for loss_name in ['focal_{}'.format(i) for i in [ 4 ]]:
                            for ctc_decoder in ['greedy' ]:
                                for remove_target_duplicates in [0 ]:
                                    for task in ['']:
                                        if 'Swin' in ecogmappingname:
                                            modelinds = 1
                                        else:
                                            modelinds = 1
                                        for trainsubject in allsub:
                                             #, 'NY688','NY836' HB
                                                MODEL_IND = 1
                                                #trainsubject =  allsub[0]
                                                testsubject =  ','.join(depth_samples) #allsub[0]
                                                
                                                #print (allsubj_param['Subj'][trainsubject]['Gender'])
                                            #for subject in [['NY717','NY742','NY749','NY798']]:
                                                n_features = 10e6
                                                batch_size = 64 // len(trainsubject.split(','))
                                                for subject in testsubject.split(','):
                                                    #print ('subject',subject)
                                                    with h5py.File('/scratch/xc1490/ECoG_Shared_Data/LD_data_extracted/meta_data/{}.h5'.format(subject),'r') as hf:
                                                        if use_mixed_signal =='None':
                                                            n_features_ = hf['ecog_alldataset'].shape[1] 
                                                        else:
                                                            n_features_ = hf['ecog_alldataset'].shape[1]*2
                                                        #print ('n_features',n_features)
                                                    n_features = min(n_features, n_features_)
                                                
                                                #if #a2a not exists:
                                                load_sub_dir ='None'# 'output_new/a2a_1005/a2a_10050800_a2a_corpus_sub_{}_nsample_80_nfft_{}_noisedb_-50_density_LD_formantsup_1_wavebased_1_bgnoisefromdata_1_load_0_ft_1_learnfilter_1_reverse_1_dynamic_0'.format(trainsubject,n_fft )
                                                #if os.path.exists(load_sub_dir):
                                                    #if len([i.split('.')[0].split('_')[1] for i in os.listdir(load_sub_dir) if i.endswith('pth')]) >0:
                                                count += 1
                                                #print (np.array([i.split('.')[0].split('_')[1][5:] for i in os.listdir(load_sub_dir) if i.endswith('pth')]).astype('int').max())
                                                output_dir = '{}_12141800'.format(ecogmappingname)#+ ecogmappingname

                                                sample = trainsubject
                                                tmpdir = '/scratch/xc1490/projects/ecog/hubert_decoding/output/{}_12141800_e_4000_cau_0_train_{}_test_{}_ta__bs_{}_ctc_greedy_filter_0_gradcl_0_extra_up_0_rmdup_0_nout_101_align_0_infer_0_reg_0_nfeature_{}_mixsignal_{}_padaud_{}_loss_name{}/'.format(ecogmappingname,sample,sample,batch_size,n_features,use_mixed_signal,pad_audio, loss_name)
                                                if os.path.exists(tmpdir):

                                                    if len(os.listdir(tmpdir))<10:
                                                        #print (sample,  os.listdir(tmpdir))
                                                        generate_sbatch_ecog_encoder_singularity_lossweight_LD_multi(ecogmappingname,\
                output_dir,ld_loss_weight, alpha_loss_weight,consonant_loss_weight,\
                RNN_LAYER=MODEL_IND,batch_size=batch_size,task=task,mapping_layers=mapping_layer, \
                region_index=region_index,multiscale=multiscale,trainsubject = trainsubject,\
                testsubject = testsubject, epoch_num=epoch_num,pretrained_model_dir = load_sub_dir,loss_name=loss_name,\
                         hidden_dim=hidden_dim,ctc_decoder=ctc_decoder,filter_dirty_sample=filter_dirty_sample,job_abbrev='multi',\
                            gradient_clip=0,extra_up=extra_up,remove_target_duplicates=remove_target_duplicates,n_outputs=n_outputs,
                         hour=12,pad_audio=pad_audio,n_features=n_features,use_mixed_signal=use_mixed_signal,align_loss=align_loss)
                                                else:
                                                    generate_sbatch_ecog_encoder_singularity_lossweight_LD_multi(ecogmappingname,\
                output_dir,ld_loss_weight, alpha_loss_weight,consonant_loss_weight,\
                RNN_LAYER=MODEL_IND,batch_size=batch_size,task=task,mapping_layers=mapping_layer, \
                region_index=region_index,multiscale=multiscale,trainsubject = trainsubject,loss_name=loss_name,\
                testsubject = testsubject, epoch_num=epoch_num,pretrained_model_dir = load_sub_dir,job_abbrev='multi',\
                         hidden_dim=hidden_dim,ctc_decoder=ctc_decoder,filter_dirty_sample=filter_dirty_sample,\
                            gradient_clip=0,extra_up=extra_up,remove_target_duplicates=remove_target_duplicates,n_outputs=n_outputs,
                             hour=12,pad_audio=pad_audio,n_features=n_features,use_mixed_signal=use_mixed_signal,align_loss=align_loss)
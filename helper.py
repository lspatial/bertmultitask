from smart_pytorch import SMARTLoss, kl_loss, sym_kl_loss
import torch.nn.functional as F
import os

def smart_regularizer(smart_wei, embeddings, logits, eval_fn,type='class'):
    """
    This function calculate the SMART regularization score.
    """
    if type == 'class':
        smart_loss_fn = SMARTLoss(eval_fn = eval_fn, loss_fn = kl_loss, loss_last_fn = sym_kl_loss,
                            num_steps = 1,          # Number of optimization steps to find noise (default = 1)
                            step_size = 1e-5,       # Step size to improve noise (default = 1e-3)
                            epsilon = 1e-6,         # Noise norm constraint (default = 1e-6)
                            noise_var = 1e-6        # Initial noise variance (default = 1e-5)
        )
    elif type == 'reg':
        smart_loss_fn = SMARTLoss(eval_fn=eval_fn, loss_fn=F.mse_loss, loss_last_fn=F.mse_loss,
                                  num_steps=1,  # Number of optimization steps to find noise (default = 1)
                                  step_size=1e-5,  # Step size to improve noise (default = 1e-3)
                                  epsilon=1e-6,  # Noise norm constraint (default = 1e-6)
                                  noise_var=1e-6  # Initial noise variance (default = 1e-5)
                                  )
    #Compute SMART loss
    smart_score = smart_wei * smart_loss_fn(embeddings, logits)
    return smart_score

def retrievePath(args,other=None):
    tpath = (args.root + '/multitasks.ly' + str(args.n_hidden_layers) + '.hz' + str(args.hidden_size) + '.bz' +
             str(args.batch_size) +'.' + args.fine_tune_mode + '.dp' + str(args.hidden_dropout_prob) + '.lr' + str(args.lr) + (
                 '_'+args.gradient_p if args.gradient_p is not None else ''))
    if hasattr(args, 'no_train_cpal') and not args.no_train_cpal:
        tpath +='_pal'
    if other is not None:
        tpath += '.'+other
    if os.path.exists(tpath):
        subfolders = [ os.path.basename(f) for f in os.scandir(tpath) if f.is_dir() and os.path.basename(f).isdigit()]
        if len(subfolders) > 0:
            res = [eval(i) for i in subfolders]
            maxres = max(res)
            tpath +=  '/' + str(maxres+1)
        else:
            tpath +=  '/' + str(1)
    else:
        tpath += '/1'
    os.makedirs(tpath, exist_ok=True)
    os.makedirs(tpath + '/predictions/', exist_ok=True)
    return tpath


def retrievePathold(args,other=None):
    tpath = (args.root + '/multitasks.ly' + str(args.n_hidden_layers) + '.hz' + str(args.hidden_size) + '.bz' +
             str(args.batch_size) +'.' + args.fine_tune_mode + '.dp' + str(args.hidden_dropout_prob) + '.lr' + str(args.lr) + (
                 '_surgery' if args.gradient_surgery else ''))
    if not args.no_train_cpal:
        tpath +='_pal'
    if other is not None:
        tpath += '.'+other
    if os.path.exists(tpath):
        subfolders = [ os.path.basename(f) for f in os.scandir(tpath) if f.is_dir() and os.path.basename(f).isdigit()]
        if len(subfolders) > 0:
            res = [eval(i) for i in subfolders]
            maxres = max(res)
            tpath +=  '/' + str(maxres+1)
        else:
            tpath +=  '/' + str(1)
    else:
        tpath += '/1'
    os.makedirs(tpath, exist_ok=True)
    os.makedirs(tpath + '/predictions/', exist_ok=True)
    return tpath

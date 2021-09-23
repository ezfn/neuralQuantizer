from quantizers import EncoderMultiDecoder
import pytorch_lightning as pl
import torch
import torchmetrics

class EncoderMultiClassifier(EncoderMultiDecoder.EncoderMultiDecoder):
    def __init__(self, encoder, decoder, primary_loss, n_embed=1024, decay=0.8, commitment=1., eps=1e-5):
        super().__init__( encoder, decoder, primary_loss, n_embed=n_embed, decay=decay, commitment=commitment, eps=eps)

        self.train_acc_handlers = []
        self.val_acc_handlers = []
        for k in range(len(self.decoder) + 1):
            self.train_acc_handlers.append(torchmetrics.Accuracy())
            self.val_acc_handlers.append(torchmetrics.Accuracy())

    def on_fit_start(self) -> None:
        super().on_fit_start()
        for handler in self.train_acc_handlers + self.val_acc_handlers:
            handler.to(self.device)

    def ensemble_calculator(self, preds_list):
        return torch.mean( torch.stack( preds_list ), axis=0 )

    def training_step(self, batch, batch_idx):
        result_dict = super().training_step(batch, batch_idx)
        accs = self.calculate_acc(result_dict['preds'], result_dict['gts'], self.train_acc_handlers, activation=torch.nn.Softmax())
        acc_dict = {}
        for idx in range( len( self.decoder ) ):
            acc_dict[f'model_{idx}_train'] = accs[idx]
        acc_dict["ensemble_train"] = accs[-1]
        self.logger.experiment.add_scalars('acc_step', acc_dict,
                                            global_step=self.global_step )

        # for idx in range(len(self.decoder)):
        #     self.log( f'train_acc_{idx}', accs[idx] )
        # self.log( f'train_acc_ensemble', accs[-1] )
        return result_dict

    def training_epoch_end(self, outputs_dicts) -> None:
        super().training_epoch_end(outputs_dicts)
        acc_dict = {}
        for idx in range( len( self.decoder ) ):
            acc_dict[f'model_{idx}_train'] = self.train_acc_handlers[idx].compute()
        acc_dict["ensemble_train"] = self.train_acc_handlers[-1].compute()
        self.logger.experiment.add_scalars( 'acc_epoch', acc_dict,
                                            global_step=self.global_step )
        # for idx in range( len( self.decoder)):
        #     self.log( f'train_acc_{idx}_epoch', self.train_acc_handlers[idx].compute())
        # self.log( f'train_acc_ensemble_epoch', self.train_acc_handlers[-1].compute())


    def validation_step(self, batch, batch_idx):
        result_dict = super().validation_step( batch, batch_idx )
        accs = self.calculate_acc( result_dict['preds'], result_dict['gts'], self.val_acc_handlers, activation=torch.nn.Softmax())
        # acc_dict = {}
        # for idx in range( len( self.decoder ) ):
        #     acc_dict[f'model_{idx}_val'] = accs[idx]
        # acc_dict["ensemble_val"] = accs[-1]
        # self.logger.experiment.add_scalars('acc_step', acc_dict,
        #                                     global_step=self.global_step )
        return result_dict



    def validation_epoch_end(self, outputs_dicts) -> None:
        super().validation_epoch_end( outputs_dicts )
        acc_dict = {}
        for idx in range( len( self.decoder ) ):
            acc_dict[f'model_{idx}_val'] = self.val_acc_handlers[idx].compute()
        acc_dict["ensemble_val"] = self.val_acc_handlers[-1].compute()
        self.logger.experiment.add_scalars('acc_epoch', acc_dict,
                                           global_step=self.global_step)

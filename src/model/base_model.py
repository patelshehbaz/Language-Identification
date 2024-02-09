import torch.nn as nn
import torch.nn.functional as F
import torch


# The `ImageClassification` class is a subclass of `nn.Module` and it defines the training and
# validation steps, and the epoch end function
class ImageClassification(nn.Module):

    def training_step(self, batch):
      """
      The `training_step` function is called once for each batch of data. It takes a batch of data as
      input, makes a prediction using the model, calculates the loss, and returns the loss.

      Returns:
        The loss
      """
      images, labels = batch 
      out = self(images)
      loss = F.cross_entropy(out, labels)
      return loss
    
    def validation_step(self, batch):
      """
      The validation_step function is called once for each batch of validation data. It takes a batch of
      data as input, runs it through the model, calculates the loss and accuracy, and returns a dictionary
      of metrics
      
      Args:
        batch: The batch of data that was passed to the validation_step function.
      
      Returns:
        The validation loss and accuracy.
      """
      images, labels = batch 
      out = self(images)
      loss = F.cross_entropy(out, labels)
      acc = accuracy(out, labels)          
      return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
      """
      The validation_epoch_end function is called at the end of each epoch, and it takes the outputs of
      the validation_step function as input. It then calculates the average loss and accuracy across all
      batches in the epoch, and returns them as a dictionary
      
      Args:
        outputs: list of dictionaries containing the outputs of the validation step.
      
      Returns:
        The validation loss and accuracy for the epoch.
      """
      batch_losses = [x['val_loss'] for x in outputs]
      epoch_loss = torch.stack(batch_losses).mean()  
      batch_accs = [x['val_acc'] for x in outputs]
      epoch_acc = torch.stack(batch_accs).mean()     
      return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
      print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
          epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
  """
  It takes the outputs of the model and the actual labels, and returns the accuracy of the model
  
  Args:
    outputs: the output of the model
    labels: The actual labels of the images.
  
  Returns:
    The accuracy of the model
  """
  _, preds = torch.max(outputs, dim=1)
  return torch.tensor(torch.sum(preds == labels).item() / len(preds))
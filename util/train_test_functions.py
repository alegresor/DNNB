''' Utility Function to train and test a dnnb classification model '''

from time import time
from pandas import DataFrame
from numpy import argmax, inf, zeros, diag, array, concatenate, isfinite


def train_test_model(model, train_loader, test_loader, max_epochs=1, max_batches=inf,
                    classes=array([0,1]), track_batch=True, predict=lambda y_hat:argmax(y_hat,1)):
    print('\nTrain & Test: '+model.name)
    epoch_fs = 'Epoch %-5d Final_Train_Loss: %-10.4f Avg_Test_Loss: %-10.4f Avg_Test_Accuracy: %-10.4f'
    batch_fs = '\t%s Batch %-5d Loss: %-10.3f Accuracy: %-10.3f'
    df_batch_metrics = DataFrame(
        columns = ['Batch','Train_Loss','RunTime','Train_Accuracy'] +
                  ['Precision_%d'%c for c in classes] + 
                  ['Recall_%d'%c for c in classes],
        dtype = float)
    df_epoch_metrics = DataFrame(
        columns = ['Epoch','Final_Train_Loss','Avg_Test_Loss','Final_Train_Accuracy','Avg_Test_Accuracy'] +
                  ['Avg_Precision_%d'%c for c in classes] + 
                  ['Avg_Recall_%d'%c for c in classes],
        dtype = float)
    batch = 0
    for epoch in range(max_epochs):
        # Training
        for j,(x,y) in enumerate(iter(train_loader)):
            t0 = time()
            batch += 1
            if batch>max_batches:
                print()
                return df_batch_metrics,df_epoch_metrics
            try:
                x = x.numpy()
                y = y.numpy()
            except:
                pass
            y = y.reshape((-1,1))
            y_hat = model.forward(x)
            train_loss = model.backward(y_hat,y)
            y_hat_digit = predict(y_hat).flatten()
            y = y.flatten()
            train_batch_accuracy = (y_hat_digit==y).sum()/len(y)
            if track_batch: 
                cm_batch_j = zeros((len(classes),len(classes)))
                for ii,jj in zip(y.astype(int),y_hat_digit.astype(int)):
                    cm_batch_j[ii,jj] += 1
                a_totals = cm_batch_j.sum()
                p_totals = cm_batch_j.sum(0)
                p_totals[p_totals==0] = inf
                r_totals = cm_batch_j.sum(1)
                r_totals[r_totals==0] = inf
                train_batch_accuracy = diag(cm_batch_j).sum() / a_totals
                train_batch_precisions = diag(cm_batch_j) / p_totals
                train_batch_recalls = diag(cm_batch_j) / r_totals
                train_batch_row = concatenate(
                    (array([batch,train_loss,time()-t0,train_batch_accuracy]),
                    train_batch_precisions,
                    train_batch_recalls))
                df_batch_metrics.loc[batch] = train_batch_row
                print(batch_fs%('Train',j+1,train_loss,train_batch_accuracy))          
        # Testing
        total_test_loss = 0.
        cm_epoch_j = zeros((len(classes),len(classes)))
        for j,(x,y) in enumerate(iter(test_loader)):
            try:
                x = x.numpy()
                y = y.numpy()
            except:
                pass
            y = y.reshape((-1,1))
            y_hat = model.forward(x)
            test_loss = model.loss_obj.f(y_hat,y)
            total_test_loss += test_loss
            y_hat_digit = predict(y_hat).flatten()
            y = y.flatten()
            if track_batch:
                test_accuracy = (y_hat_digit==y).sum()/len(y)
                print(batch_fs%('Test',j+1,test_loss,test_accuracy))
            for ii,jj in zip(y.astype(int),y_hat_digit.astype(int)):
                cm_epoch_j[ii,jj] += 1
        # Epoch's Metrics
        avg_test_loss = total_test_loss/(j+1)
        a_totals = cm_epoch_j.sum()
        p_totals = cm_epoch_j.sum(0)
        p_totals[p_totals==0] = inf
        r_totals = cm_epoch_j.sum(1)
        r_totals[r_totals==0] = inf
        avg_test_accuracy = diag(cm_epoch_j).sum() / a_totals
        avg_test_precisions = diag(cm_epoch_j) / p_totals
        avg_test_recalls = diag(cm_epoch_j) / r_totals
        avg_test_precisions[~ isfinite(avg_test_precisions)] = 0 # account for divide by 0
        avg_test_recalls[~ isfinite(avg_test_recalls)] = 0 # account for divide by 0
        epoch_row = concatenate(
            (array([epoch+1,train_loss,avg_test_loss,train_batch_accuracy,avg_test_accuracy]),
            avg_test_precisions,
            avg_test_recalls))
        df_epoch_metrics.loc[epoch+1] = epoch_row         
        print(epoch_fs%(epoch+1,train_loss,avg_test_loss,avg_test_accuracy))
    print()
    return df_batch_metrics,df_epoch_metrics
    
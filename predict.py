import torch
from config import args

TASKS = [param for param in args.pk_params]

def test(model, loader, device):
    model.eval()

    preds = {task: [] for task in TASKS}
    att_info = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose, orig_dose = [item.to(device) for item in batch]

            # scores.shape: (batch_size, num_subs)
            y_hat, scores = model(mol_gdata, sub_gdata, sub_mask, token_ids, token_mask, dose)
            
            # Store predictions for this fold by task
            for mol_idx, dose_item, y_hat_item in zip(range(len(mol_gdata)),
                                                      orig_dose.detach().cpu(),
                                                      y_hat.detach().cpu()):
                for task_idx, task in enumerate(TASKS):
                    preds[task].append({
                        'smiles': mol_gdata[mol_idx].smi,
                        'dose': list(dose_item.unsqueeze(0).numpy())[0],
                        'y_hat': list(y_hat_item[task_idx].unsqueeze(0).numpy())[0]
                    })
            
            # substructure-level attention scores
            for mol_idx in range(scores.size(0)):
                sub_indices = (sub_mask[mol_idx] == 1).nonzero(as_tuple=True)[0].tolist()
                
                sub_scores = []
                for sub_idx in sub_indices:
                    sub_scores.append({
                        'sub_index': sub_idx,
                        'sub_smiles': sub_gdata[sub_idx].smi,
                        'sub_score': list(scores[mol_idx, sub_idx].detach().cpu().unsqueeze(0).numpy())[0]
                    })
            
                att_info.append({
                    'smiles': mol_gdata[mol_idx].smi,
                    'dose': list(orig_dose[mol_idx].detach().cpu().unsqueeze(0).numpy())[0],
                    'sub_scores': sub_scores
                })

    return preds, att_info

if __name__ == '__main__':
    pass
    
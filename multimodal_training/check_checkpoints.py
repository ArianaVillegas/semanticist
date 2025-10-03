import torch

# Check OLD checkpoint
print("OLD checkpoint (coco_small_fixed):")
ckpt1 = torch.load('checkpoints/coco_small_fixed/best_model.pt', map_location='cpu')
print(f"  Epoch: {ckpt1['epoch']}")
print(f"  Best loss: {ckpt1.get('best_loss', 'N/A')}")
if 'args' in ckpt1:
    args = ckpt1['args']
    print(f"  Lambda caption: {getattr(args, 'lambda_caption', 'N/A')}")
print()

# Check NEW checkpoint
print("NEW checkpoint (coco_test_retraining):")
ckpt2 = torch.load('checkpoints/coco_test_retraining/best_model.pt', map_location='cpu')
print(f"  Epoch: {ckpt2['epoch']}")
print(f"  Best loss: {ckpt2.get('best_loss', 'N/A')}")
if 'args' in ckpt2:
    args = ckpt2['args']
    print(f"  Lambda caption: {getattr(args, 'lambda_caption', 'N/A')}")
print()

# Check if they're the same
same_weights = True
for key in ckpt1['model_state_dict'].keys():
    if not torch.equal(ckpt1['model_state_dict'][key], ckpt2['model_state_dict'][key]):
        same_weights = False
        break

print(f"Checkpoints have identical weights: {same_weights}")

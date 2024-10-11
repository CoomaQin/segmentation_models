import os  
import torch  
import torch.distributed as dist  

def main():  
    # Initialize the process group  
    dist.init_process_group(backend='nccl')  

    # Set the local GPU device  
    local_rank = int(os.environ['LOCAL_RANK'])  
    torch.cuda.set_device(local_rank)  

    # Create a tensor  
    tensor = torch.ones(10).cuda()  

    # Perform all-reduce  
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  

    print(f"Rank {dist.get_rank()} has tensor: {tensor}")  

    # Clean up  
    dist.destroy_process_group()  

if __name__ == "__main__":  
    main() 
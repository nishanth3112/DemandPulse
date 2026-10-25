# Tearing down the AWS resources

The project keeps its data in one private S3 bucket and uses one IAM user. When you are done, remove
them so nothing keeps billing and no access keys remain. **This is destructive and cannot be
undone.** Nothing here has been run for you; read each step first.

What you keep afterwards: the code, the docs, and `demandpulse demo`, which needs no AWS. If you
want the dashboard to keep working on the real results, run `demandpulse app export` first. The
snapshot it writes to `data/app/` is about 5 MB and the dashboard reads only that.

Replace the placeholders. The pipeline's own IAM user is limited to the bucket, so steps 2 and 3
need an administrator identity, not the `demandpulse` profile.

## 1. Delete the bucket

```bash
export AWS_PROFILE=demandpulse AWS_REGION=us-east-1
BUCKET=<value of DEMANDPULSE_S3_BUCKET>

aws s3 ls "s3://$BUCKET" --recursive --summarize | tail -3     # what is there (about 7 GB)
aws s3api get-bucket-versioning --bucket "$BUCKET"             # empty output means unversioned
aws s3 rb "s3://$BUCKET" --force                               # delete every object, then the bucket
```

If versioning is `Enabled`, `rb --force` leaves old versions behind and the bucket will not
delete. Use the console instead: S3, select the bucket, Empty, then Delete.

## 2. Delete the IAM user and its keys

Using an administrator profile (replace `<admin>`):

```bash
USER=demandpulse-pipeline
aws iam list-access-keys --user-name "$USER" --profile <admin>
aws iam delete-access-key --user-name "$USER" --access-key-id <each key id> --profile <admin>
aws iam list-user-policies --user-name "$USER" --profile <admin>
aws iam delete-user-policy --user-name "$USER" --policy-name <name> --profile <admin>
aws iam delete-user --user-name "$USER" --profile <admin>
```

Earlier in the project AWS keys were pasted into a chat session twice. If any key from that time
still exists, deleting the user's keys here removes it. Check the list before deleting the user.

## 3. Delete the budget, if you created one

```bash
aws budgets describe-budgets --account-id <ACCOUNT_ID> --profile <admin>
aws budgets delete-budget --account-id <ACCOUNT_ID> --budget-name <name> --profile <admin>
```

## 4. Check nothing is left

```bash
aws s3api list-buckets --profile <admin>
aws iam list-users --profile <admin>
```

Look at Billing and Cost Explorer a day later, because charges post with a delay.

## 5. Local cleanup (optional)

- Remove the `demandpulse` profile from `~/.aws/credentials` and `~/.aws/config`.
- Delete generated local data: `data/` (it is not in Git).
- Remove the Docker image: `docker rmi demandpulse:dev`.

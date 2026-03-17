# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: study-club (3.11.14)
#     language: python
#     name: python3
# ---

# %%
from seq2seq.train import main

main()

# %%
for epoch in range(200):
    iterator = tqdm.tqdm(loader(dataset), total=len(dataset))
    total_loss = 0

    for data, label in iterator:
        data = torch.tensor(data, dtype=torch.long).to(device)
        label = torch.tensor(label, dtype=torch.long).to(device)

        # 인코더의 초기 은닉 상태
        encoder_hidden = torch.zeros(1, 1, 64).to(device)
        # 인코더의 모든 시점의 출력을 저장하는 변수
        encoder_outputs = torch.zeros(11, 64).to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0
        for ei in range(len(data)):
            # ➊한 단어씩 인코더에 넣어줌
            encoder_output, encoder_hidden = encoder(data[ei], encoder_hidden)
            # ❷인코더의 은닉 상태를 저장
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[0]]).to(device)

        # ❸인코더의 마지막 은닉 상태를 디코더의 초기 은닉 상태로 저장
        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < 0.5 else False  # ❶

        if use_teacher_forcing:
            for di in range(len(label)):
                decoder_output = decoder(decoder_input, decoder_hidden, encoder_outputs)

                # 직접적으로 정답을 다음 시점의 입력으로 넣어줌
                target = torch.tensor(label[di], dtype=torch.long).to(device)
                target = torch.unsqueeze(target, dim=0).to(device)
                loss += nn.CrossEntropyLoss()(decoder_output, target)
                decoder_input = target
        else:
            for di in range(len(label)):
                decoder_output = decoder(decoder_input, decoder_hidden, encoder_outputs)

                # ➊ 가장 높은 확률을 갖는 단어의 인덱스가 topi
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()

                # 디코더의 예측값을 다음 시점의 입력으로 넣어줌
                target = torch.tensor(label[di], dtype=torch.long).to(device)
                target = torch.unsqueeze(target, dim=0).to(device)
                loss += nn.CrossEntropyLoss()(decoder_output, target)

                if decoder_input.item() == 1:  # <EOS> 토큰을 만나면 중지
                    break
        # 전체 손실 계산
        total_loss += loss.item() / len(dataset)
        iterator.set_description(f'epoch:{epoch + 1} loss:{total_loss}')
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

torch.save(encoder.state_dict(), 'attn_enc.pth')
torch.save(decoder.state_dict(), 'attn_dec.pth')

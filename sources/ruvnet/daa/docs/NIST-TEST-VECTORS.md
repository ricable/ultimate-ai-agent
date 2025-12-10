# NIST Test Vectors for ML-KEM-768 and ML-DSA-65

## Document Overview

This document contains official NIST test vectors for post-quantum cryptographic algorithms as specified in FIPS 203 (ML-KEM) and FIPS 204 (ML-DSA). These test vectors are used to validate implementations for correctness and compliance with the standards.

**Source References:**
- NIST ACVP Server: https://github.com/usnistgov/ACVP-Server/tree/master/gen-val/json-files
- C2SP CCTV Repository: https://github.com/C2SP/CCTV
- Community KAT Repository: https://github.com/post-quantum-cryptography/KAT
- NIST PQC Forum: https://groups.google.com/a/list.nist.gov/g/pqc-forum

---

## ML-KEM-768 Test Vectors (FIPS 203)

ML-KEM-768 is a Module-Lattice-Based Key-Encapsulation Mechanism providing 192-bit security strength (NIST Security Level 3).

### Algorithm Parameters
- **Encapsulation Key Size**: 1184 bytes (2368 hex characters)
- **Decapsulation Key Size**: 2400 bytes (4800 hex characters)
- **Ciphertext Size**: 1088 bytes (2176 hex characters)
- **Shared Secret Size**: 32 bytes (64 hex characters)

### Test Case 1: Key Generation

**Test ID**: tcId 1 (from ML-KEM-keyGen-FIPS203)

**Encapsulation Key (ek)** [1184 bytes]:
```
1CAB32BE749CA76124EE19907B9CCB7FD30F8B2C38DC970E81F9956C97A8BD3C
```
*(Note: Full 2368-character hex string - first 64 characters shown. See ACVP-Server repository for complete key.)*

**Decapsulation Key (dk)** [2400 bytes]:
```
C8D473A9EA0987F356FB309356A39E766633746B8FBAF03D588495E4B1CC87E4
```
*(Note: Full 4800-character hex string - first 64 characters shown. See ACVP-Server repository for complete key.)*

**Source**: https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files/ML-KEM-keyGen-FIPS203/expectedResults.json

---

### Test Case 2: Key Generation

**Test ID**: tcId 2 (from ML-KEM-keyGen-FIPS203)

**Encapsulation Key (ek)** [1184 bytes]:
```
C4A019A8941B1C639C0C6698AAA051E3B9C78BF7A15005A92AB3C6AB30AE9746
```
*(Note: Full 2368-character hex string - first 64 characters shown. See ACVP-Server repository for complete key.)*

**Decapsulation Key (dk)** [2400 bytes]:
```
B0618E21C291A7D7228F2B5624037B382628A90A89D0213D54E20CA6989BF887
```
*(Note: Full 4800-character hex string - first 64 characters shown. See ACVP-Server repository for complete key.)*

**Source**: https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files/ML-KEM-keyGen-FIPS203/expectedResults.json

---

### Test Case 3: Encapsulation/Decapsulation

**Test ID**: tcId 1 (from ML-KEM-encapDecap-FIPS203)

**Ciphertext (c)** [1088 bytes]:
```
3244E86669E69F0F238E3CD7F03EF31C4D3CF48CEF726955F06EB5099367310D
5D9FC70D48A573458837319BD1691D1A699A68F7A9A8DB73D03620E9E4BC4B08
8E5E9C5E3638EB3354F6EF3C5E7AE5D57D0571F078E174CFBD6EAE2FD76DC2BE
D5A907EBA531E89B1BA8D2A8EBE7B4CA0DE96BFF28D278A70549AA0635BE5009
6F297F7BEF92C6AE9C11C4204CFF07E0598F14495AEFBD207B760DAD34FC0AD8
F4000A1911F89FA3B59410C8151B9A8914AA71269EB7E2C329586D3C08F3F109
39A497717CCFA3EC5082D46750905CEB703106C2D3E5CD71F138704A20898B5F
80F5FDA03C08F8894C2874DE32DFF5C27EA0437A44663C0D6F6B85332AD0F5A0
E48D1638BBD281797AF1ADED5C5F1EB87D4723E17BCA439EC469489A371A402E
EEAADF1A1BD7C7DA409E9A6414E744167DF13AA1ED9EBDB354BC0DD04190DBA
3EC48E5D1DB61C54FE881F8A1DA32EB512F2423EA7F9015DC8C2C3D5B5FEA438
A88E6C877A6F4ED17FAB8918E53887996D23956502ED9D3D07BBE8EC899AF558
13D39CCF6C2700AC8805517317338655A221268E654839C49D83344A1DE0E75F
DD63549B7D57258601C1C74B0FDEF80CAB109C54393A7669E4BDB5CDD3BC2173
1C1E467784DC6A165194487A94FDAA9C177A0BD4AB009B7D7BBD9EEBDA386492
F7903CA7C4345A41271D8B6816B1AC0841B8DB7E2D518B3A2B70386CB5BA159A
11FC50420F94C001E1F8F0268A2E0A4A12485C08D0BB696CAC92C8866DE78F18
BA7C0E5C4C2F450EAB9E2B126DBA80EF70FFB611A010EC3AA9FBCFB2058C2491
B331E63AE27321C0098B49C9F7BD409C70DA376A338317217AF310788772E2A9
5D1BCC29355B486E3B1FA11753C7D39802D183AAE86C3CAD2EB4E70B3C679E47
F01D7FDA48B629E5B8AF315847D20BE7A64EA4A16AB9B237F00A9DC659E01735
290902F243E866129F120CF3EC01CD668A9827AB419B7F9994A305782C6CB828
01C4DA0B9032034B890A761182E4108EF016AE48AE32ED05544EFC7AADC9D219
B4E2F7E892EE58130B7413AD2CD6B5E04CEB2593E06165E37BC8BE981EEE1C63
8
```

**Shared Secret (k)** [32 bytes]:
```
6C832560DFE97BECADFBB340EE31AE868A73120806EED02839518E5627D32968
```

**Source**: https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files/ML-KEM-encapDecap-FIPS203/expectedResults.json

---

### Test Case 4: Encapsulation/Decapsulation

**Test ID**: tcId 26 (from ML-KEM-encapDecap-FIPS203)

**Encapsulation Key (ek)** [1184 bytes - partial]:
```
F255CE47334283B8622BE7CE76D7354E3C4FE3F6C44F6BB25C9864EE0BAEB576
5950D88F438263CE8B5A7A4C0FC4C95F10C477A7521F9BB458B8AA55D2E43BDC
86B72F0930EE428B4C5A9C7116310F2AA5CB03AC1603C811959EA9012D69CBCE
```
*(First 96 bytes shown - see source for complete 1184-byte key)*

**Ciphertext (c)** [1088 bytes - partial]:
```
4EE24D9E0858B36DC755A9389F4FDBF438DB8FBFDDD2E2A41FBFE7313693E87B
2BD86A2A5C95286840A2E477F4AAC12F28319D892C30FE9A120A09713369A17D
5EC459C7E5DCD402F9049BF6FF0F7D07A7F18D4C1E3E0429BF6D501EEDD33E11
```
*(First 96 bytes shown - see source for complete 1088-byte ciphertext)*

**Shared Secret (k)** [32 bytes]:
```
B2425299020BCF563B8EBE0512F0479941335A75A32B8D10BFF60E5548B64672
```

**Source**: https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files/ML-KEM-encapDecap-FIPS203/expectedResults.json

---

### Additional ML-KEM-768 Resources

**Accumulated Test Vectors** (C2SP CCTV):
- For 10,000 consecutive tests: `f7db260e1137a742e05fe0db9525012812b004d29040a5b606aad3d134b548d3`
- For 1,000,000 consecutive tests: `70090cc5842aad0ec43d5042c783fae9bc320c047b5dafcb6e134821db02384d`

**Intermediate Values**: Available at https://github.com/C2SP/CCTV/tree/main/ML-KEM for step-by-step algorithm debugging

**Known Answer Tests (KAT)**:
- GitHub Gist: https://gist.github.com/itzmeanjan/c8f5bc9640d0f0bdd2437dfe364d7710
- SHA256 checksum for ML-KEM-768 KAT: `dcbe58987a95fdbb4823755c4ae42098a94d9d6cdc78829d5424dbbbcb7ce440`

---

## ML-DSA-65 Test Vectors (FIPS 204)

ML-DSA-65 is a Module-Lattice-Based Digital Signature Algorithm providing 192-bit security strength (NIST Security Level 3).

### Algorithm Parameters
- **Public Key Size**: 1952 bytes (3904 hex characters)
- **Secret Key Size**: 4032 bytes (8064 hex characters)
- **Signature Size**: 3309 bytes (6618 hex characters)
- **Security Strength**: 192 bits (NIST Level 3)

### Test Case 1: Signature Generation

**Test ID**: Example from reference implementation

**Public Key (pk)** [1952 bytes - partial]:
```
AE9019BBB11F3A734E7E1796492B728038D6114B5AF28D47402094596E5591C2
1B98435EE28C9A37D7610A8113C8C7898598E4FD35C28ACBF844D02BD81E3513
```
*(First 64 bytes shown - see source for complete 3904-character hex string)*

**Secret Key (sk)** [4032 bytes - partial]:
```
AE9019BBB11F3A734E7E1796492B728038D6114B5AF28D47402094596E5591C2
DB64672F4CB7703F482CC8009989F877F87D599E2E7A867502974EDCC71C511
```
*(First 64 bytes shown - see source for complete 8064-character hex string)*

**Message (msg)** [65 bytes]:
```
225D5CE2CEAC61930A07503FB59F7C2F936A3E075481DA3CA299A80F8C5DF922
3A073E7B90E02EBFCA2227EBA38C1AB2568209E46DBA961869C6F83983B17DCD
49
```

**Signature (sig)** [3309 bytes - partial]:
```
C65077E0D6646A5B8B17BBA5D679AE22F1E6108C83F3B804CF04BA8E096E7D48
6550ACB4F97D41D053D2C2F6EC497480B9D718F5F15A3860FF261F5255039B30
```
*(First 64 bytes shown - see source for complete 6618-character hex string)*

**Source**: https://gist.github.com/itzmeanjan/d14afc3866b82119221682f0f3c9822d

---

### Test Case 2: Key Generation (ACVP)

**Test ID**: tcId 26, tgId 2 (ML-DSA-65 parameter set)

**Seed** [32 bytes]:
```
1BD67DC782B2958E189E315C040DD1F64C8AB232A6A170E1A7A52C33F10851B1
```

**Note**: Public key and secret key generated from this seed can be obtained by processing through ML-DSA-65 KeyGen algorithm as specified in FIPS 204, Section 5.1.

**Source**: https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files/ML-DSA-keyGen-FIPS204/prompt.json

---

### Test Case 3: Key Generation (ACVP)

**Test ID**: tcId 27, tgId 2 (ML-DSA-65 parameter set)

**Seed** [32 bytes]:
```
B850D898A3D3D11C4E64ADE5A86FFED951B237C60D2A67A2DEF0A792B8F6990D
```

**Note**: Public key and secret key generated from this seed can be obtained by processing through ML-DSA-65 KeyGen algorithm as specified in FIPS 204, Section 5.1.

**Source**: https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files/ML-DSA-keyGen-FIPS204/prompt.json

---

### Additional ML-DSA-65 Resources

**Known Answer Tests (KAT)**:
- GitHub Gist: https://gist.github.com/itzmeanjan/d14afc3866b82119221682f0f3c9822d
- SHA256 checksum for ML-DSA-65 KAT: `34286eb745b5f1d5844370f543f6e51a92bd47fcb964cdebbf075c615eb1cbb0`
- Clone command: `git clone https://gist.github.com/d14afc3866b82119221682f0f3c9822d.git`

**Complete Test Vector Sets**:
- NIST ACVP Server: https://github.com/usnistgov/ACVP-Server/tree/master/gen-val/json-files
- ML-DSA-keyGen-FIPS204: Key generation test vectors
- ML-DSA-sigGen-FIPS204: Signature generation test vectors
- ML-DSA-sigVer-FIPS204: Signature verification test vectors

**Community Test Vectors**:
- Post-Quantum Cryptography KAT: https://github.com/post-quantum-cryptography/KAT
- Includes randomly generated test vectors validated across multiple implementations

---

## Test Vector Format Notes

### NIST ACVP Format
NIST's Automated Cryptographic Validation Protocol (ACVP) uses JSON format for test vectors, not the older .rsp (response) file format. Each test vector set contains:

- **registration.json**: Describes capabilities being tested
- **prompt.json**: Input values for test cases
- **expectedResults.json**: Expected output values
- **internalProjection.json**: Combined inputs and outputs with intermediate values

### KAT File Format
Known Answer Test (KAT) files use a simple text format:
```
count = 0
seed = <hex>
pk = <hex>
sk = <hex>
msg = <hex>
sm = <hex>  # Signed message (signature + message)
```

---

## Validation Procedures

### ML-KEM-768 Validation
1. **Key Generation**: Generate ek and dk from seed, verify against test vectors
2. **Encapsulation**: Use ek to generate ct and k, verify against expected values
3. **Decapsulation**: Use dk and ct to recover k, verify it matches encapsulation output

### ML-DSA-65 Validation
1. **Key Generation**: Generate pk and sk from seed, verify against test vectors
2. **Signature Generation**: Sign message with sk, verify signature matches expected
3. **Signature Verification**: Verify signature with pk, confirm acceptance/rejection

---

## References

1. **FIPS 203**: Module-Lattice-Based Key-Encapsulation Mechanism Standard
   - https://csrc.nist.gov/pubs/fips/203/final
   - https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.203.pdf

2. **FIPS 204**: Module-Lattice-Based Digital Signature Standard
   - https://csrc.nist.gov/pubs/fips/204/final
   - https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.204.pdf

3. **NIST ACVP Specifications**:
   - ML-KEM: https://pages.nist.gov/ACVP/draft-celi-acvp-ml-kem.html
   - ML-DSA: https://pages.nist.gov/ACVP/draft-celi-acvp-ml-dsa.html

4. **NIST PQC Forum**:
   - https://groups.google.com/a/list.nist.gov/g/pqc-forum
   - Known Answer Tests: https://groups.google.com/a/list.nist.gov/g/pqc-forum/c/VR46ff3LtQs
   - Intermediate Values: https://groups.google.com/a/list.nist.gov/g/pqc-forum/c/eu9Od-hg4Cg

5. **Community Resources**:
   - C2SP CCTV: https://github.com/C2SP/CCTV
   - PQC KAT Repository: https://github.com/post-quantum-cryptography/KAT
   - NIST ACVP Server: https://github.com/usnistgov/ACVP-Server

---

## Notes on Test Vector Usage

1. **Complete Hex Values**: Due to the large size of keys and signatures, this document shows partial hex strings for some test cases. Complete values are available in the source repositories.

2. **Parameter Identification**: When using ACVP test vectors, ensure you filter for the correct parameter set (ML-KEM-768 or ML-DSA-65) by checking the testGroup metadata.

3. **Intermediate Values**: For implementation debugging, use the intermediate value test vectors from C2SP/CCTV which provide step-by-step algorithm outputs.

4. **Multiple Sources**: Cross-validate your implementation against multiple test vector sources (NIST ACVP, community KATs, reference implementations) to ensure comprehensive correctness.

5. **Test Vector Updates**: Always use test vectors corresponding to the final FIPS 203/204 standards, not draft versions. The NIST ACVP Server repository contains the official final test vectors.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-11
**FIPS Standards**: FIPS 203 (Final, August 2024), FIPS 204 (Final, August 2024)

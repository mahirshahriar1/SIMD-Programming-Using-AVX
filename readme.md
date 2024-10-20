# AVX Instructions Reference

This reference provides a starting point for using AVX (Advanced Vector Extensions) instructions, organized by category. instructions. For more detailed information, refer to the [Intel Intrinsics Guide or official documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)

AVX is designed to improve performance for applications that can take advantage of parallel processing, especially in tasks involving large data sets.

## AVX Integer Instructions

### Addition
- **Add Packed 8-bit Integers**
  - **Function**: `_mm256_add_epi8`
  - **Description**: Adds packed 8-bit integers, saturating if overflow occurs.
  - **Example**:
    ```cpp
    __m256i result = _mm256_add_epi8(a, b);
    ```
  
- **Add Packed 16-bit Integers**
  - **Function**: `_mm256_add_epi16`
  - **Description**: Adds packed 16-bit integers, saturating if overflow occurs.
  - **Example**:
    ```cpp
    __m256i result = _mm256_add_epi16(a, b);
    ```
  
- **Add Packed 32-bit Integers**
  - **Function**: `_mm256_add_epi32`
  - **Description**: Adds packed 32-bit integers.
  - **Example**:
    ```cpp
    __m256i result = _mm256_add_epi32(a, b);
    ```
  
- **Add Packed 64-bit Integers**
  - **Function**: `_mm256_add_epi64`
  - **Description**: Adds packed 64-bit integers.
  - **Example**:
    ```cpp
    __m256i result = _mm256_add_epi64(a, b);
    ```

### Subtraction
- **Subtract Packed 8-bit Integers**
  - **Function**: `_mm256_sub_epi8`
  - **Description**: Subtracts packed 8-bit integers, saturating if underflow occurs.
  - **Example**:
    ```cpp
    __m256i result = _mm256_sub_epi8(a, b);
    ```
  
- **Subtract Packed 16-bit Integers**
  - **Function**: `_mm256_sub_epi16`
  - **Description**: Subtracts packed 16-bit integers, saturating if underflow occurs.
  - **Example**:
    ```cpp
    __m256i result = _mm256_sub_epi16(a, b);
    ```
  
- **Subtract Packed 32-bit Integers**
  - **Function**: `_mm256_sub_epi32`
  - **Description**: Subtracts packed 32-bit integers.
  - **Example**:
    ```cpp
    __m256i result = _mm256_sub_epi32(a, b);
    ```
  
- **Subtract Packed 64-bit Integers**
  - **Function**: `_mm256_sub_epi64`
  - **Description**: Subtracts packed 64-bit integers.
  - **Example**:
    ```cpp
    __m256i result = _mm256_sub_epi64(a, b);
    ```

### Multiplication
- **Multiply Packed 16-bit Integers**
  - **Function**: `_mm256_mullo_epi16`
  - **Description**: Multiplies packed 16-bit integers, returning lower 16 bits of each product.
  - **Example**:
    ```cpp
    __m256i result = _mm256_mullo_epi16(a, b);
    ```
  
- **Multiply Packed 32-bit Integers**
  - **Function**: `_mm256_mullo_epi32`
  - **Description**: Multiplies packed 32-bit integers, returning lower 32 bits of each product.
  - **Example**:
    ```cpp
    __m256i result = _mm256_mullo_epi32(a, b);
    ```

### Bitwise Operations
- **Bitwise AND**
  - **Function**: `_mm256_and_si256`
  - **Description**: Performs a bitwise AND operation on packed integers.
  - **Example**:
    ```cpp
    __m256i result = _mm256_and_si256(a, b);
    ```

- **Bitwise OR**
  - **Function**: `_mm256_or_si256`
  - **Description**: Performs a bitwise OR operation on packed integers.
  - **Example**:
    ```cpp
    __m256i result = _mm256_or_si256(a, b);
    ```

- **Bitwise XOR**
  - **Function**: `_mm256_xor_si256`
  - **Description**: Performs a bitwise XOR operation on packed integers.
  - **Example**:
    ```cpp
    __m256i result = _mm256_xor_si256(a, b);
    ```

- **Bitwise NOT**
  - **Function**: `_mm256_not_si256`
  - **Description**: Performs a bitwise NOT operation (using `_mm256_andnot_si256`).
  - **Example**:
    ```cpp
    __m256i result = _mm256_andnot_si256(a, b);
    ```

### Comparison
- **Compare Greater Than (Packed 8-bit Integers)**
  - **Function**: `_mm256_cmpgt_epi8`
  - **Description**: Compares packed 8-bit integers for greater-than and returns a mask.
  - **Example**:
    ```cpp
    __m256i result = _mm256_cmpgt_epi8(a, b);
    ```
  
- **Compare Greater Than (Packed 32-bit Integers)**
  - **Function**: `_mm256_cmpgt_epi32`
  - **Description**: Compares packed 32-bit integers for greater-than and returns a mask.
  - **Example**:
    ```cpp
    __m256i result = _mm256_cmpgt_epi32(a, b);
    ```

## AVX Floating Point Instructions

### Addition
- **Add Packed Single-Precision Floating-Point Values**
  - **Function**: `_mm256_add_ps`
  - **Description**: Adds packed single-precision floating-point values.
  - **Example**:
    ```cpp
    __m256 result = _mm256_add_ps(a, b);
    ```

### Subtraction
- **Subtract Packed Single-Precision Floating-Point Values**
  - **Function**: `_mm256_sub_ps`
  - **Description**: Subtracts packed single-precision floating-point values.
  - **Example**:
    ```cpp
    __m256 result = _mm256_sub_ps(a, b);
    ```

### Multiplication
- **Multiply Packed Single-Precision Floating-Point Values**
  - **Function**: `_mm256_mul_ps`
  - **Description**: Multiplies packed single-precision floating-point values.
  - **Example**:
    ```cpp
    __m256 result = _mm256_mul_ps(a, b);
    ```

### Division
- **Divide Packed Single-Precision Floating-Point Values**
  - **Function**: `_mm256_div_ps`
  - **Description**: Divides packed single-precision floating-point values.
  - **Example**:
    ```cpp
    __m256 result = _mm256_div_ps(a, b);
    ```

### Square Root
- **Square Root of Packed Single-Precision Floating-Point Values**
  - **Function**: `_mm256_sqrt_ps`
  - **Description**: Computes the square root of packed single-precision floating-point values.
  - **Example**:
    ```cpp
    __m256 result = _mm256_sqrt_ps(a);
    ```

## Additional Instructions

### Load/Store Instructions
- **Load Unaligned 256-bit Integer Values**
  - **Function**: `_mm256_loadu_si256`
  - **Description**: Loads unaligned 256-bit integer values from memory.
  - **Example**:
    ```cpp
    __m256i result = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    ```

- **Store Unaligned 256-bit Integer Values**
  - **Function**: `_mm256_storeu_si256`
  - **Description**: Stores unaligned 256-bit integer values to memory.
  - **Example**:
    ```cpp
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), result);
    ```

### Miscellaneous
- **Horizontal Addition**
  - **Function**: `_mm256_hadd_ps`
  - **Description**: Horizontally adds adjacent pairs of packed single-precision floating-point values.
  - **Example**:
    ```cpp
    __m256 result = _mm256_hadd_ps(a, b);
    ```



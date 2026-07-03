import struct

def demonstrate_bit_punning(original_float, payload_bits, num_lsb=2):
    print("=" * 60)
    print(f" STARTING VALUE: {original_float} (Type: {type(original_float).__name__})")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Pack the float into raw C-style bytes
    # -------------------------------------------------------------------------
    packed_float = struct.pack(">f", original_float)
    
    # Let's convert those bytes into a neat string of 32 binary bits (1s and 0s)
    bits_32 = "".join(f"{b:08b}" for b in packed_float)
    
    print("\n[STEP 1] Packing float into a Python bytes object...")
    print(f"-> Python bytes representation : {packed_float}")
    print(f"-> Hexadecimal representation  : 0x{packed_float.hex().upper()}")
    print(f"-> Raw 32-bit binary string    : {bits_32}")
    
    # Visually separating the IEEE 754 fields: Sign (1), Exponent (8), Mantissa (23)
    sign = bits_32[0]
    exponent = bits_32[1:9]
    mantissa = bits_32[9:]
    print("\n--- IEEE 754 Field Breakdown ---")
    print(f" Sign bit (1 bit)       : {sign}  (0 = positive, 1 = negative)")
    print(f" Exponent (8 bits)      : {exponent} (Decimal equivalent: {int(exponent, 2)})")
    print(f" Mantissa (23 bits)     : {mantissa}")
    print("--------------------------------")

    # -------------------------------------------------------------------------
    # STEP 2: Unpack the bytes as an unsigned integer (Type Punning)
    # -------------------------------------------------------------------------
    int_representation = struct.unpack(">I", packed_float)[0]
    
    print("\n[STEP 2] Unpacking those exact same bits as a C Unsigned Int...")
    print(f"-> Resulting Python integer    : {int_representation}")
    print(f"-> Integer binary layout       : {int_representation:032b}")
    print("   (Notice how the integer binary perfectly matches the raw float binary above!)")

    # -------------------------------------------------------------------------
    # STEP 3: Manipulate the bits using masks (Data Hiding / Steganography)
    # -------------------------------------------------------------------------
    print(f"\n[STEP 3] Injecting payload ({payload_bits}) into the lowest {num_lsb} bits...")
    
    # Create the bitmask (e.g., if num_lsb is 2, mask is 0000...0011)
    mask = (1 << num_lsb) - 1
    
    # ~mask flips all bits (e.g., 1111...1100). ANDing clears the target bits to 0.
    cleared_int = int_representation & (~mask)
    
    # ORing injects our payload data into those freshly cleared slots
    new_int_representation = cleared_int | (payload_bits & mask)
    
    print(f"-> Bitmask used                : {mask:032b}")
    print(f"-> Cleaned integer binary      : {cleared_int:032b}")
    print(f"-> Modified integer binary     : {new_int_representation:032b}")
    print(f"-> Modified integer value      : {new_int_representation}")

    # -------------------------------------------------------------------------
    # STEP 4: Pack the modified integer back and unpack it as a float
    # -------------------------------------------------------------------------
    repacked_bytes = struct.pack(">I", new_int_representation)
    modified_float = struct.unpack(">f", repacked_bytes)[0]
    
    print("\n[STEP 4] Packing modified int back to bytes, then unpacking as a float...")
    print(f"-> Modified float value        : {modified_float}")
    print(f"-> Absolute numeric change     : {modified_float - original_float}")
    print("=" * 60)

# Run the demonstration with our example: 1.5
# We will inject the binary value '3' (binary 11) into the lowest 2 bits.
demonstrate_bit_punning(original_float=1.5, payload_bits=3, num_lsb=2) 
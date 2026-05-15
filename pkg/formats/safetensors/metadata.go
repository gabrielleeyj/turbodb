package safetensors

import "strconv"

// TurboQuant metadata keys recorded under the SafeTensors __metadata__ map so
// exported quantized indexes are self-describing and portable.
const (
	MetaVersion     = "turboquant_version"
	MetaRotatorSeed = "rotator_seed"
	MetaRotatorType = "rotator_type"
	MetaCodebookID  = "codebook_id"
	MetaBitWidth    = "bit_width"
	MetaVariant     = "variant"
)

// CurrentVersion is the TurboQuant metadata schema version written by this
// package.
const CurrentVersion = "0.1"

// QuantMeta describes the quantization parameters of an exported index. It
// converts to and from the flat string map SafeTensors stores.
type QuantMeta struct {
	Version     string
	RotatorSeed uint64
	RotatorType string // e.g. "hadamard"
	CodebookID  string // e.g. "d1536_b4_lloyd_max_v1"
	BitWidth    int
	Variant     string // "mse" or "prod"
}

// ToMap renders the quantization metadata as SafeTensors string metadata.
func (m QuantMeta) ToMap() map[string]string {
	version := m.Version
	if version == "" {
		version = CurrentVersion
	}
	return map[string]string{
		MetaVersion:     version,
		MetaRotatorSeed: strconv.FormatUint(m.RotatorSeed, 10),
		MetaRotatorType: m.RotatorType,
		MetaCodebookID:  m.CodebookID,
		MetaBitWidth:    strconv.Itoa(m.BitWidth),
		MetaVariant:     m.Variant,
	}
}

// ParseQuantMeta extracts quantization metadata from a SafeTensors metadata
// map. Missing numeric fields default to zero; malformed numbers return an
// error so callers can reject corrupt files.
func ParseQuantMeta(md map[string]string) (QuantMeta, error) {
	m := QuantMeta{
		Version:     md[MetaVersion],
		RotatorType: md[MetaRotatorType],
		CodebookID:  md[MetaCodebookID],
		Variant:     md[MetaVariant],
	}
	if s := md[MetaRotatorSeed]; s != "" {
		seed, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			return QuantMeta{}, err
		}
		m.RotatorSeed = seed
	}
	if s := md[MetaBitWidth]; s != "" {
		bw, err := strconv.Atoi(s)
		if err != nil {
			return QuantMeta{}, err
		}
		m.BitWidth = bw
	}
	return m, nil
}

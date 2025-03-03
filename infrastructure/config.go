package infrastructure

type Config struct {
	PythonURL string
}

func LoadConfig() Config {
	return Config{
		PythonURL: "http://localhost:5000",
	}
}

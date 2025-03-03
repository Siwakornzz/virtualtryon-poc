package http_client

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"virtualtryon/domain/entities"
	"virtualtryon/infrastructure"
)

type PythonClient struct {
	url string
}

func NewPythonClient() *PythonClient {
	config := infrastructure.LoadConfig()
	return &PythonClient{url: config.PythonURL}
}

func (c *PythonClient) ProcessImage(image entities.Image, clothing entities.Clothing) (entities.Image, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, _ := writer.CreateFormFile("image", image.Filename)
	part.Write(image.Data)

	part, _ = writer.CreateFormFile("clothing", clothing.Filename)
	part.Write(clothing.Data)

	writer.Close()

	resp, err := http.Post(c.url+"/process", writer.FormDataContentType(), body)
	if err != nil {
		return entities.Image{}, err
	}
	defer resp.Body.Close()

	result := entities.Image{Filename: "result.png"}
	result.Data, err = io.ReadAll(resp.Body)
	if err != nil {
		return entities.Image{}, err
	}
	return result, nil
}

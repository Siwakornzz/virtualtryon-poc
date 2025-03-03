package main

import (
	"io"
	"virtualtryon/adapters/http_client"
	"virtualtryon/domain/entities"
	"virtualtryon/domain/services"

	"github.com/gofiber/fiber/v2"
)

func main() {
	app := fiber.New()

	pythonClient := http_client.NewPythonClient()
	tryOnService := services.NewTryOnService(pythonClient)

	app.Post("/upload", func(c *fiber.Ctx) error {
		imageFile, err := c.FormFile("image")
		if err != nil {
			return c.Status(400).SendString("Failed to get image file")
		}
		clothingFile, err := c.FormFile("clothing")
		if err != nil {
			return c.Status(400).SendString("Failed to get clothing file")
		}

		imageData, err := imageFile.Open()
		if err != nil {
			return c.Status(500).SendString("Failed to open image")
		}
		defer imageData.Close()

		clothingData, err := clothingFile.Open()
		if err != nil {
			return c.Status(500).SendString("Failed to open clothing")
		}
		defer clothingData.Close()

		imageBytes, err := io.ReadAll(imageData)
		if err != nil {
			return c.Status(500).SendString("Failed to read image data")
		}
		clothingBytes, err := io.ReadAll(clothingData)
		if err != nil {
			return c.Status(500).SendString("Failed to read clothing data")
		}

		image := entities.Image{Filename: imageFile.Filename, Data: imageBytes}
		clothing := entities.Clothing{Filename: clothingFile.Filename, Data: clothingBytes}

		result, err := tryOnService.TryOn(image, clothing)
		if err != nil {
			return c.Status(500).SendString(err.Error())
		}

		// Send as PNG
		c.Set("Content-Type", "image/png")
		return c.Send(result.Data)
	})

	app.Listen(":3000")
}

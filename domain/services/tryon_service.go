package services

import (
	"virtualtryon/domain/entities"
	"virtualtryon/domain/ports"
)

type TryOnService struct {
	processor ports.ImageProcessor
}

func NewTryOnService(processor ports.ImageProcessor) *TryOnService {
	return &TryOnService{processor: processor}
}

func (s *TryOnService) TryOn(image entities.Image, clothing entities.Clothing) (entities.Image, error) {
	return s.processor.ProcessImage(image, clothing)
}

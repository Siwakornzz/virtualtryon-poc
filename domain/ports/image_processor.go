package ports

import "virtualtryon/domain/entities"

type ImageProcessor interface {
	ProcessImage(image entities.Image, clothing entities.Clothing) (entities.Image, error)
}

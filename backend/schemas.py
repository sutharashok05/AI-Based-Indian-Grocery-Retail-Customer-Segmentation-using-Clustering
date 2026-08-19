from pydantic import BaseModel, Field


class ProductInput(BaseModel):

    price: float = Field(
        ge=0
    )

    discount: float = Field(
        ge=0
    )

    rating: float = Field(
        ge=0,
        le=5
    )

    reviews: int = Field(
        ge=0
    )

    title_length: int = Field(
        ge=0
    )

    feature_length: int = Field(
        ge=0
    )

    description_length: int = Field(
        ge=0
    )
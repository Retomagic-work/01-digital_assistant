import { IsNotEmpty } from "class-validator";

export class CreateUsersDto {
  @IsNotEmpty()
  login: string;

  @IsNotEmpty()
  password: string;
}
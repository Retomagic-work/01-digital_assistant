import {
  Column,
  Entity,
  PrimaryGeneratedColumn,
  Index,
  CreateDateColumn,
  UpdateDateColumn
} from 'typeorm';
import {RequestsStatusInterface} from "../interfaces/requests.status.interface";
import {RequestsDataDto} from "../dto/requests-data.dto";
import {RabbitmqRequestsDto} from "../../rabbitmq/dto/rabbitmq-requests.dto";
import {RequestsResponseDataDto} from "../dto/requests-response-data.dto";

@Entity('requests')
export class RequestEntity {
  @Index({unique: true})
  @PrimaryGeneratedColumn()
  id: number;

  @Index()
  @Column({default: null, nullable: true})
  rating: number;

  @Index()
  @Column({default: RequestsStatusInterface.pending})
  status: RequestsStatusInterface;

  @Column({
    default: [],
    type: 'text',
    array: true
  })
  original_images: string[]

  @Column({
    default: [],
    type: 'text',
    array: true
  })
  finished_images: string[]

  @Column({
    default: {},
    type: 'jsonb',
  })
  data: RequestsDataDto

  @Column({
    default: {},
    type: 'jsonb',
  })
  response: RequestsResponseDataDto

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}
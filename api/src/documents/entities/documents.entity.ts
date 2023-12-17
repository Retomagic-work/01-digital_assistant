import {
  Column,
  Entity,
  PrimaryGeneratedColumn,
  Index,
  CreateDateColumn,
  UpdateDateColumn
} from 'typeorm';

@Entity('documents')
export class DocumentsEntity {
  @Index({unique: true})
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  description: string;

  @Column()
  link: string;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;
}